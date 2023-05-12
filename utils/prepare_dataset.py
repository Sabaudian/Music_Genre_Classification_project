import os
import math

from pydub import AudioSegment
from pydub.utils import make_chunks


def makedir(dir_path):
    # create a new directory
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print("\n Folder " + dir_path + "has been crated successfully!")
        
        
def check_file_extension(dataset_path, file_extension):

    print("\n Checking audio file extension...")

    # loop through all genre sub-folder
    for i, (dirpath, dirname, filenames) in enumerate(os.walk(dataset_path)):

        # loop through audio file
        for f in sorted(filenames):

            # to ignore a hidden file in my folder
            if not f.startswith("."):

                # pick file extension
                current_extension = f.split(".")[-1]  # -> wav
                # file path
                file_path = os.path.join(dirpath, f)
                sound = AudioSegment.from_file(file_path)

                # check extension
                if current_extension != file_extension:
                    print("File {} is not .{}".format(f, file_extension))
                    sound.export(file_path, format=file_extension)

    print("\n ...all audio file has been checked!")


def check_sound_duration(dataset_path, milliseconds_duration):

    print("\n Checking audio file duration...")

    # loop through all genre sub-folder
    for i, (dirpath, dirname, filenames) in enumerate(os.walk(dataset_path)):

        # ensure we're processing a genre sub-folder level
        if dirpath is not dataset_path:

            # loop through audio file
            for f in sorted(filenames):

                # pick file extension
                extension = f.split(".")[-1]

                # control the duration -> at least milliseconds_duration (30000 ms)
                file_path = os.path.join(dirpath, f)
                sound = AudioSegment.from_file(file_path, format=extension)
                duration_in_milliseconds = len(sound)

                # if it is less than my threshold add silence
                if duration_in_milliseconds < milliseconds_duration:

                    # pick file with duration less milliseconds_duration
                    print("\n- File: \033[92m{}\033[0m, Duration: \033[92m{}\033[0m".format(f, duration_in_milliseconds))
                    # compute duration difference between my threshold and sound duration
                    duration_difference = milliseconds_duration - len(sound)
                    # append silence to reach the minimum quota
                    new_file_duration = sound[:len(sound)] + AudioSegment.silent(duration=duration_difference)
                    print("\n-- Adjust Duration to: \033[92m{}\033[0m".format(len(new_file_duration)))
                    # export file
                    new_file_duration.export(file_path, format=extension)

    print("\n ...all audio file has been checked!")


# divides each audio file into 3-second long files
def make_chunks_from_data(dataset_path, chunk_length, new_dir_path):

    if not os.path.isdir(new_dir_path):
        # make new directory for pre-processed data
        makedir(new_dir_path)

        print("\nData Augmentation in progress...\n")

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
                # make new sub-dir for every genre
                makedir(new_dir_path + "/" + semantic_label)

                # process all audio files in genre sub-folder
                for f in sorted(filenames):

                    # pick full file format
                    print("\n- File: \033[92m{}\033[0m".format(f))

                    # pick file extension
                    file_extension = f.split(".")[-1]  # -> wav
                    file_id = f.split(".")[-2]  # -> 0000
                    file_genre = f.split(".")[-3]  # -> blues
                    genre_plus_id = file_genre + "." + file_id  # -> blues.0000

                    # pick current file path
                    current_file_path = os.path.join(dirpath, f)
                    print("- Location: \033[92m{}\033[0m".format(current_file_path))

                    if os.path.isfile(current_file_path):
                        print("- Location is File: \033[92m{}\033[0m".format(os.path.isfile(current_file_path)))

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
                            output_path = str(new_dir_path) + "/" + str(semantic_label) + "/" + str(chunk_name)
                            chunk.export(output_path, format=file_extension)
        print("\n...all data has been processed!\n")
    else:
        print("\n The data has already been processed! Proceed with the features extraction")


