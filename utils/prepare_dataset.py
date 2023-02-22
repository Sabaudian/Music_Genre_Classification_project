import os
import math
import shutil

from pydub import AudioSegment
from pydub.utils import which
from pydub.utils import make_chunks


def check_sound_duration(dataset_path):
    # loop through all genre sub-folder
    for i, (dirpath, dirname, filenames) in enumerate(os.walk(dataset_path)):

        # ensure we're processing a genre sub-folder level
        if dirpath is not dataset_path:

            # loop through audio file
            for f in sorted(filenames):

                # pick file extension
                extension = f.split(".")[-1]

                # control the duration -> at lest 30000 ms
                file_path = os.path.join(dirpath, f)
                sound = AudioSegment.from_file(file_path, format=extension)
                duration_in_milliseconds = len(sound)
                if duration_in_milliseconds < 30000:
                    # pick file with wrong duration
                    print("\n- File: \033[92m{}\033[0m, Duration: \033[92m{}\033[0m".format(f, duration_in_milliseconds))
                    # compute duration difference between my threshold and sound duration
                    duration_difference = 30000 - len(sound)
                    # append silence to reach the minimum quota
                    new_file_duration = sound[:len(sound)] + AudioSegment.silent(duration=duration_difference)
                    print("\n- Adjust Duration to: \033[92m{}\033[0m".format(len(new_file_duration)))
                    # export file
                    new_file_duration.export(file_path, format=extension)


# divides each audio file into 3-second long files
def make_chunks_from_data(dataset_path, chunk_length):
    # loop through all genre sub-folder
    for i, (dirpath, dirname, filenames) in enumerate(os.walk(dataset_path)):

        # ensure we're processing a genre sub-folder level
        if dirpath is not dataset_path:
            print("\n- Dirpath: \033[92m{}\033[0m".format(dirpath))

            # saving folder root and genre label
            dirpath_root = dirpath.split('/')[-2]
            print("- Dirpath Root: \033[92m{}\033[0m".format(dirpath_root))
            semantic_label = dirpath.split("/")[-1]
            print("- Semantic Label: \033[92m{}\033[0m".format(semantic_label))

            # process all audio files in genre sub-folder
            for f in sorted(filenames):

                # pick full file format
                print("\n- File: \033[92m{}\033[0m".format(f))

                # pick file extension
                file_extension = f.split(".")[-1]
                # print("- Type: \033[92m{}\033[0m".format(file_extension))
                file_id = f.split(".")[-2]
                # print("- ID: \033[92m{}\033[0m".format(file_id))
                file_genre = f.split(".")[-3]
                # print("- Genre: \033[92m{}\033[0m".format(file_genre))
                genre_plus_id = file_genre + "." + file_id
                # print("- Genre + ID: \033[92m{}\033[0m".format(genre_plus_id))

                # pick current file path
                current_file_path = os.path.join(dirpath, f)
                print("- Location: \033[92m{}\033[0m".format(current_file_path))

                if os.path.isfile(current_file_path):
                    print("- Location is File: \033[92m{}\033[0m".format(os.path.isfile(current_file_path)))

                    original_file = os.path.join(dirpath + "/original_30s", f)
                    # print("- Original File: \033[92m{}\033[0m".format(original_file))

                    # to better specify path to interpreter
                    AudioSegment.converter = which("ffmpeg")
                    # pick information about audio file in dataset
                    my_audio_file = AudioSegment.from_file(current_file_path, format=file_extension)
                    print("\n- Sound Duration: \033[92m{}\033[0m".format(len(my_audio_file)))

                    # define chunks
                    chunks = make_chunks(my_audio_file, chunk_length=chunk_length)
                    print("- Computed Chunk: \033[92m{}\033[0m".format(len(chunks)))
                    # compute module to adjust the number of chunks
                    if math.fmod(len(my_audio_file), chunk_length) != 0:
                        chunks = chunks[:len(chunks) - 1]
                        print("- Generated Chunks: \033[92m{}\033[0m".format(len(chunks)))

                    # exporting single 3 seconds long chunks as file
                    for j, chunk in enumerate(chunks):
                        chunk_name = genre_plus_id + "_{}.".format(j) + file_extension
                        print("- Exporting: \033[92m{}\033[0m".format(chunk_name))
                        output_path = str(dirpath_root) + "/" + str(semantic_label) + "/" + str(chunk_name)
                        chunk.export(output_path, format=file_extension)

                    # move the original 30 seconds long audio file in a sub-folder
                    print("\n- Check if folder exist: \033[92m{}\033[0m".format(os.path.isdir(dirpath + "/original_30s")))
                    if not os.path.exists(dirpath + "/original_30s"):
                        os.makedirs(dirpath + "/original_30s")
                        shutil.move(current_file_path, original_file)
                    else:
                        shutil.move(current_file_path, original_file)


