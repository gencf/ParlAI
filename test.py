import os
import time
os.environ['MKL_THREADING_LAYER'] = 'GNU'
# from numba import jit, cuda

from matplotlib import pyplot as plt

def write_file(test):
    with open("train_corpus_size.txt", "r") as file:
        corpus_size = int(file.read())
    with open("test_path.txt", "r") as file:
        test_path = file.read()
    with open(test_path, "a") as file:
        file.write("Corpus_Size: {} Loss: {:.9} ({:.4}% of the dataset)\n".format(corpus_size, str(test['loss']), str(float(corpus_size/6.4))))

# @jit(target_backend ="cuda")
if __name__ == '__main__':
    ROOT_PATH = "./z"
    # FILE_NAME = "test_transformers"
    FILE_NAME = "test_image_seq2seq"
    TEST_NAME_LIST = []
    # CORPUS_SIZE_LIST = [4, 8, 32, 64, 128, 320, 640]
    CORPUS_SIZE_LIST = [640]
    NUM_TESTS = 1

    FILE_PATH = os.path.join(ROOT_PATH, FILE_NAME)
    if not os.path.exists(FILE_PATH):
        os.makedirs(FILE_PATH)

    with open(os.path.join(FILE_PATH, "test_names.txt"), "r") as file:
        test_names = file.read().splitlines()
        for test_name in test_names:
            TEST_NAME_LIST.append(test_name)

    for TEST_NAME in TEST_NAME_LIST:
        TEST_PATH = os.path.join(FILE_PATH, TEST_NAME.split("::")[0] + "_" + TEST_NAME.split("::")[1])
        if not os.path.exists(TEST_PATH):
            os.makedirs(TEST_PATH)

        TEST_PATH = os.path.join(TEST_PATH, TEST_NAME.split("::")[0] + "_" + TEST_NAME.split("::")[1])

        loss_graph = {}
        time_graph = {}

        for CORPUS_SIZE in CORPUS_SIZE_LIST:
            with open("train_corpus_size.txt", "w") as file:
                file.write(str(CORPUS_SIZE))

            with open("test_path.txt", "w") as file:
                file.write(TEST_PATH + ".txt")

            start_txt_read_time = time.time()
            with open("train_corpus_size.txt", "r") as file:
                cor = int(file.read())
            with open("test_path.txt", "r") as file:
                test_path = file.read()
            end_txt_read_time = time.time()
            txt_read_time = end_txt_read_time - start_txt_read_time
            print("TXT read time: {}".format(txt_read_time))

            start = time.time()
            for i in range(NUM_TESTS):
                os.system("pytest -v tests/{}.py::{}".format(FILE_NAME, TEST_NAME))
            end = time.time()

            average_time = (end - start) / NUM_TESTS - 2*txt_read_time
            print("Average Time: {}".format(average_time))

            with open("{}.txt".format(TEST_PATH), "r") as file:
                lines = file.readlines()
                sum = 0
                flag = 0
                num_test = 0

                for line in lines:
                    if line == "\n" or line == "" or line.startswith("Average"):
                        continue

                    else:
                        line = line.strip("\n").split()

                        if int(line[1]) == CORPUS_SIZE:
                            sum += float(line[3])
                            flag = 1
                            num_test += 1

                        elif flag == 1:
                            break

                        else:
                            continue

            print(CORPUS_SIZE, num_test)

            with open("{}.txt".format(TEST_PATH), "r+") as file:
                flag = 0

                while True:
                    line = file.readline()

                    if line == "" or line == "\n":
                        if flag == 1:
                            average_loss = sum / num_test
                            file.write("Average Loss: {:.7}\n". format(average_loss))
                            file.write("Average Time: {:.7}\n\n". format(average_time))
                            loss_graph[CORPUS_SIZE] = average_loss
                            time_graph[CORPUS_SIZE] = average_time
                            break

                        else:
                            continue   # skip empty line

                    elif line.startswith("Average"):
                        continue

                    elif not line:
                        break

                    else:
                        line = line.split()

                        if int(line[1]) == CORPUS_SIZE:
                            flag = 1

                        elif flag == 1:
                            break

                        else:
                            continue

        print(loss_graph)
        print(time_graph)
        plt.figure()
        plt.plot(list(loss_graph.keys())[:], list(loss_graph.values())[:])
        plt.xlabel("Corpus Size")
        plt.ylabel("Average Loss")
        plt.title("Average Loss vs Corpus Size")
        plt.savefig("{}_loss_graph.pdf".format(TEST_PATH))
        plt.close()

        plt.figure()
        plt.plot(time_graph.keys(), time_graph.values())
        plt.xlabel("Corpus Size")
        plt.ylabel("Average Time")
        plt.title("Average Time vs Corpus Size")
        plt.savefig("{}_time_graph.pdf".format(TEST_PATH))

