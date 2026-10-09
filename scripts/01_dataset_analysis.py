import os
import json
import logging
import pandas as pd
from pathlib import Path
from collections import Counter
from tqdm import tqdm
from mutagen.mp3 import MP3
# Configure logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
logging.basicConfig(
    filename=log_dir / "2026-07-08_analysis.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def analyze_dataset():
    logging.info("Starting dataset analysis...")
    
    meta_dir = Path("data/raw/saraga/metadata")
    audio_dir = Path("data/raw/saraga/audio")
    
    if not meta_dir.exists():
        logging.error(f"Metadata directory {meta_dir} does not exist!")
        print(f"Error: Metadata directory {meta_dir} does not exist!")
        return
        
    recordings = []
    
    # Recursively find all JSON files
    json_paths = list(meta_dir.rglob("*.json"))
    logging.info(f"Found {len(json_paths)} JSON files in metadata directory.")
    
    for json_path in tqdm(json_paths, desc="Parsing metadata files"):
        # Get relative path to find matching audio file
        rel_path = json_path.relative_to(meta_dir)
        
        # Audio file path matches the metadata relative path, but with .mp3.mp3 extension
        audio_filename = json_path.stem + ".mp3.mp3"
        audio_path = audio_dir / rel_path.parent / audio_filename
        
        # Read JSON
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception as e:
            logging.error(f"Failed to parse JSON file {json_path}: {e}")
            continue
            
        rec_data = {
            "title": meta.get("title", ""),
            "mbid": meta.get("mbid", ""),
            "json_path": str(json_path),
            "audio_path": str(audio_path) if audio_path.exists() else None,
            "has_audio": audio_path.exists(),
        }
        
        # Extract raag
        raags = meta.get("raags", [])
        if raags:
            rec_data["raag"] = raags[0].get("common_name", None)
        else:
            rec_data["raag"] = None
            
        # Extract artists and instruments
        artists = meta.get("artists", [])
        artist_names = []
        instruments = []
        lead_instrument = None
        is_vocal = False
        
        for a in artists:
            name = a.get("artist", {}).get("name", "")
            inst = a.get("instrument", {}).get("name", "")
            lead = a.get("lead", False)
            artist_names.append(name)
            instruments.append(inst)
            if lead:
                lead_instrument = inst
                if inst.lower() == "voice":
                    is_vocal = True
                    
        rec_data["artists"] = artist_names
        rec_data["instruments"] = instruments
        rec_data["lead_instrument"] = lead_instrument
        rec_data["is_vocal"] = is_vocal
        
        # Extract tonic (Sa) from the ctonic file in metadata directory
        ctonic_file = json_path.with_suffix("").with_suffix(".ctonic.txt")
        if ctonic_file.exists():
            try:
                with open(ctonic_file, "r", encoding="utf-8") as tf:
                    rec_data["tonic"] = tf.read().strip()
            except Exception as e:
                logging.warning(f"Failed to read ctonic file {ctonic_file}: {e}")
                rec_data["tonic"] = None
        else:
            rec_data["tonic"] = None
            
        # Extract duration from audio file using mutagen
        if rec_data["has_audio"]:
            try:
                audio = MP3(audio_path)
                rec_data["duration"] = audio.info.length
                rec_data["sample_rate"] = audio.info.sample_rate
                rec_data["channels"] = audio.info.channels
                rec_data["bitrate"] = audio.info.bitrate
                rec_data["file_format"] = "MP3"
            except Exception as e:
                logging.error(f"Failed to read audio properties from {audio_path}: {e}")
                rec_data["duration"] = meta.get("length", 0) / 1000.0  # Fallback to json length (ms)
                rec_data["sample_rate"] = None
                rec_data["channels"] = None
                rec_data["bitrate"] = None
                rec_data["file_format"] = "MP3"
        else:
            rec_data["duration"] = meta.get("length", 0) / 1000.0
            rec_data["sample_rate"] = None
            rec_data["channels"] = None
            rec_data["bitrate"] = None
            rec_data["file_format"] = None
            
        recordings.append(rec_data)
        
    df = pd.DataFrame(recordings)
    
    # Setup reports folder
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    
    # 1. Generate duplicate report
    # Rules: "Compare recordings using: filename, duration, artist, album, raag. Generate: duplicate_report.csv"
    # To facilitate matching, we retrieve album title from JSON files
    album_list = []
    for json_path in df["json_path"]:
        with open(json_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
            release = meta.get("release", [])
            album = release[0].get("title", "") if release else ""
            album_list.append(album)
    df["album"] = album_list
    df["filename"] = df["audio_path"].apply(lambda x: Path(x).name if x else "")
    df["artists_str"] = df["artists"].apply(lambda x: ", ".join(x))
    
    # Find duplicates
    duplicate_mask = df.duplicated(subset=["filename", "duration", "artists_str", "album", "raag"], keep=False)
    duplicates_df = df[duplicate_mask]
    
    dup_report_path = reports_dir / "duplicate_report.csv"
    duplicates_df.to_csv(dup_report_path, index=False)
    logging.info(f"Duplicate report written to {dup_report_path} with {len(duplicates_df)} potential duplicates.")
    
    # 2. Generate Dataset Statistics Report
    total_recordings = len(df)
    unique_raags = df["raag"].dropna().unique()
    raag_counts = df["raag"].value_counts()
    
    missing_raag = df["raag"].isna().sum()
    missing_audio = (~df["has_audio"]).sum()
    
    formats = df["file_format"].value_counts()
    sample_rates = df["sample_rate"].value_counts()
    channels = df["channels"].value_counts().map({1: "Mono", 2: "Stereo"})
    
    durations = df["duration"].dropna()
    
    # Tonic statistics
    has_tonic = df["tonic"].notna().sum()
    
    # Vocal vs Instrumental
    # A recording is vocal if "Voice" is in instruments
    df["vocal_inferred"] = df["instruments"].apply(lambda insts: "Voice" in insts)
    vocal_count = df["vocal_inferred"].sum()
    instrumental_count = total_recordings - vocal_count
    
    report_path = reports_dir / "dataset_statistics.txt"
    with open(report_path, "w", encoding="utf-8") as rf:
        rf.write("==================================================\n")
        rf.write("SARAGA HINDUSTANI DATASET STATISTICS REPORT\n")
        rf.write("==================================================\n\n")
        rf.write(f"Total Recordings Found: {total_recordings}\n")
        rf.write(f"Total Unique Raags: {len(unique_raags)}\n\n")
        
        rf.write("--- Raag Distribution ---\n")
        for raag, count in raag_counts.items():
            rf.write(f"  {raag}: {count}\n")
        rf.write("\n")
        
        rf.write("--- Missing Labels/Files ---\n")
        rf.write(f"  Missing Raag Labels: {missing_raag}\n")
        rf.write(f"  Missing Audio Files: {missing_audio}\n\n")
        
        rf.write("--- Audio Properties ---\n")
        rf.write(f"  Formats: {dict(formats)}\n")
        rf.write(f"  Sample Rates: {dict(sample_rates)}\n")
        rf.write(f"  Channels: {dict(channels)}\n\n")
        
        rf.write("--- Duration Distribution (Seconds) ---\n")
        rf.write(f"  Mean: {durations.mean():.2f}s\n")
        rf.write(f"  Std Dev: {durations.std():.2f}s\n")
        rf.write(f"  Min: {durations.min():.2f}s\n")
        rf.write(f"  25%: {durations.quantile(0.25):.2f}s\n")
        rf.write(f"  50% (Median): {durations.median():.2f}s\n")
        rf.write(f"  75%: {durations.quantile(0.75):.2f}s\n")
        rf.write(f"  Max: {durations.max():.2f}s\n\n")
        
        rf.write("--- Metadata Availability ---\n")
        rf.write(f"  Raag Field Included: Yes\n")
        rf.write(f"  Artist Field Included: Yes\n")
        rf.write(f"  Tonic (Sa) Included: Yes (available for {has_tonic}/{total_recordings} recordings)\n")
        rf.write(f"  Instruments Included: Yes\n")
        rf.write(f"  Vocal/Instrumental Split (Inferred from Voice instrument):\n")
        rf.write(f"    Vocal: {vocal_count}\n")
        rf.write(f"    Instrumental: {instrumental_count}\n")
        
    logging.info(f"Dataset statistics report written to {report_path}")
    print(f"Dataset analysis completed!")
    print(f"Stats report: {report_path}")
    print(f"Duplicate report: {dup_report_path}")

if __name__ == "__main__":
    analyze_dataset()
