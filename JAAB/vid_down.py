from pytubefix import YouTube
from pytubefix.exceptions import VideoUnavailable
import csv
import subprocess
import os

# List of boxing videos
video_data = [
    ["Ben Whittaker v Leon Willings", "1920x1080", "25", "20VAlk9xzWw"],
    ["Fabio Wardley v Frazer Clarke", "1920x1080", "25", "zRcJMCOSjZU"],
    ["Jessica McCaskill v Lauren Price Highlights", "1920x1080", "25", "_EIdO22GLYs"],
    ["Ben Whittaker v Ezra Arenyeka Fight Highlights", "1920x1080", "25", "y8oomIWNh_A"],
    ["Chris Billam-Smith v Richard Riakporhe 2", "1920x1080", "25", "rI940LD-GTA"],
    ["Caroline Dubois v Maira Moneo Fight Highlights", "1920x1080", "25", "K8P0MQFU3rI"],
    ["Zak Chelli v Callum Simpson Fight Highlights", "1920x1080", "25", "cTP1emraD-s"],
    ["Adam Azim v Ohara Davies", "1920x1080", "25", "edUQSh6hgjg"],
    ["Demolition Job! Callum Simpson v Elvis Ahorgah", "1920x1080", "25", "_lpHet5wnu8"],
    ["MASTERCLASS! Adam Azim v Sergei Lipinets", "1920x1080", "25", "9wnTrg-8MEI"],
    ["Caroline Dubois v Bo Me Ri Shin", "1920x1080", "25", "biMKM0g7Y0U"],
    ["Joshua Buatsi vs Dan Azeez", "1920x1080", "25", "zlPey5wMdA0"],
    ["Ben Whittaker vs Khalid Graidia", "1920x1080", "25", "PnEaz6s-zj0"],
    ["Lawrence Okolie v Chris Billam-Smith", "1920x1080", "25", "WyCXYdbpPCY"],
    ["Sam Eggington v Joe Pigford", "1920x1080", "25", "ebCT31hqr0U"],
    ["Viddal Riley v Anees Taj", "1920x1080", "25", "69yMkxIwLbI"],
    ["Viddal Riley vs Anees Taj 2", "1920x1080", "25", "aze74n2Nq78"],
    ["Caroline Dubois vs Yanina Lescano", "1920x1080", "25", "qYbCSwHzqNk"],
    ["Frazer Clarke vs Mariusz Wach", "1920x1080", "25", "_dxevohXYJE"],
    ["Callum Simpson vs Celso Neves", "1920x1080", "25", "kFTfdkiOHKo"],
    ["Ben Whittaker vs Jordan Grant", "1920x1080", "25", "X8nwkG8-_Nk"],
    ["Savannah Marshall vs Femke Hermans", "1920x1080", "25", "JOSsRxIDYqg"],
    ["Ben Whittaker vs Vladimir Belujsky", "1920x1080", "25", "RTXn9tnhfFE"],
    ["Savannah Marshall vs Franchon Crews-Dezurn", "1920x1080", "25", "ARXtIW402xM"],
    ["Liam Smith vs Chris Eubank Jr 2", "1920x1080", "25", "EhfoCfLn0-A"],
    ["Florian Marku vs Dylan Moran", "1920x1080", "25", "9Ec8NryAEj8"],
    ["Viddal Riley vs Nathan Quarless", "1920x1080", "25", "yAxpvLWHg4s"],
    ["Claressa Shields vs Savannah Marshall", "1920x1080", "25", "Q7VO3tQMBR4"],
    ["Mikael Lawal vs Isaac Chamberlain", "1920x1080", "25", "NTrkWqj9TpA"],
    ["Richard Riakporhe vs Dylan Bregeon", "1920x1080", "25", "Tqi8v8spZb8"],
    ["Franck Petitjean v Adam Azim", "1920x1080", "25", "5Ey-oHX-qto"],
    ["Tyler Denny vs Matteo Signani", "1920x1080", "25", "99TqpIp1wrI"],
    ["Ben Whittaker vs Stiven Leonetti Dredhaj", "1920x1080", "25", "bdn4KD4IzEY"],
    ["Chris Billam Smith vs Mateusz Masternak", "1920x1080", "25", "Ffe3pcjhOK4"],
    ["Natasha Jonas v Mikaela Mayer", "1920x1080", "25", "yQpyjg9gTQU"],
]

# Define file paths
output_dir = "/mnt/hdd_2t/T-DEED/boxing_dataset"
csv_file = "/mnt/hdd_2t/T-DEED/youtube_down_boxing_meta.csv"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Open CSV file and start processing
with open(csv_file, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["name", "resolution", "fps", "yt_id", "status"])

    for name, resolution, fps, yt_id in video_data:
        url = f"https://www.youtube.com/watch?v={yt_id}"
        try:
            yt = YouTube(url)

            # Get 1080p video-only and audio streams
            video_stream = yt.streams.filter(res="1080p", progressive=False, file_extension="mp4").first()
            audio_stream = yt.streams.filter(only_audio=True, file_extension="mp4").first()

            if video_stream and audio_stream:
                print(f"Downloading {name} in 1080p...")

                # Define file paths
                video_path = f"{output_dir}/{yt_id}_video.mp4"
                audio_path = f"{output_dir}/{yt_id}_audio.mp4"
                final_path = f"{output_dir}/{yt_id}.mp4"

                # Download video and audio separately
                video_stream.download(filename=f"{name}_vid.mp4", output_path="/mnt/hdd_2t/T-DEED/boxing_dataset")
                audio_stream.download(filename=f"{name}_aud.mp4", output_path="/mnt/hdd_2t/T-DEED/boxing_dataset")

                # Merge using ffmpeg
                merge_cmd = f"ffmpeg -y -i {video_path} -i {audio_path} -c:v copy -c:a aac {final_path}"
                subprocess.run(merge_cmd, shell=True, check=True)

                # Remove temp files
                os.remove(video_path)
                os.remove(audio_path)

                writer.writerow([name, resolution, fps, yt_id, "Downloaded"])
            else:
                print(f"1080p unavailable for {name}. Downloading best available.")
                yt.streams.get_highest_resolution().download(output_path="/mnt/hdd_2t/T-DEED/boxing_dataset", filename=f"{name}_vid.mp4")
                writer.writerow([name, resolution, fps, yt_id, "Downloaded (Lower Res)"])

        except VideoUnavailable:
            print(f"Video {yt_id} is unavailable.")
            writer.writerow([name, resolution, fps, yt_id, "Unavailable"])
        except Exception as e:
            print(f"Error downloading {name}: {e}")
            writer.writerow([name, resolution, fps, yt_id, f"Failed: {e}"])

print(f"Download complete. Metadata saved at: {csv_file}")
