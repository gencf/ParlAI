import os
import time
os.environ['MKL_THREADING_LAYER'] = 'GNU'

from matplotlib import pyplot as plt

if __name__ == '__main__':
    ROOT_PATH = "./z"
    FILE_NAME = "test_transformers"
    # FILE_NAME = "test_image_seq2seq"
    TEST_NAME_LIST = []
    CORPUS_SIZE_LIST = [4, 8, 32, 64, 128, 320, 640]

    FILE_PATH = os.path.join(ROOT_PATH, FILE_NAME)
    if not os.path.exists(FILE_PATH):
        os.makedirs(FILE_PATH)

    with open(os.path.join(FILE_PATH, "test_names.txt"), "r") as file:
        test_names = file.read().splitlines()
        for test_name in test_names:
            TEST_NAME_LIST.append(test_name)

    # TEST_NAME_LIST = ["TestDecoderOnly::test_train"]
    # TEST_NAME_LIST = ["TestTransformerRanker::test_alt_reduction"]

    for TEST_NAME in TEST_NAME_LIST:
        TEST_PATH = os.path.join(FILE_PATH, TEST_NAME.split("::")[0] + "_" + TEST_NAME.split("::")[1])
        if not os.path.exists(TEST_PATH):
            os.makedirs(TEST_PATH)

        TEST_PATH = os.path.join(TEST_PATH, TEST_NAME.split("::")[0] + "_" + TEST_NAME.split("::")[1])

        loss_graph = {}
        time_graph = {}

        for CORPUS_SIZE in CORPUS_SIZE_LIST:
            with open("{}.txt".format(TEST_PATH), "r") as file:
                flag = 0

                while True:
                    line = file.readline()

                    if line == "" or line == "\n" :
                        continue

                    elif line.startswith("Average Loss"):
                        if flag == 1:
                            loss_graph[CORPUS_SIZE] = float(line.split()[-1])
                        else:
                            continue
                    
                    elif line.startswith("Average Time"):
                        if flag == 1:
                            time_graph[CORPUS_SIZE] = float(line.split()[-1])
                            break
                        else:
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
        plt.plot(list(loss_graph.keys())[2:], list(loss_graph.values())[2:])
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

