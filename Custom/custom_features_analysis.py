import random

def picklines(thefile, whatlines):
    return [x for i, x in enumerate(thefile) if i in whatlines]

def main():
    """ all one main """
    path = "/home/dchouren/Documents/Princeton_Courses/COS424/HW1/"
    incorrect_ind_path = path + "custom_incorrect_ind.txt"
    correct_ind_path = path + "custom_correct_ind.txt"

    csv = open(path + "custom_features_test.csv", 'r')

    incorrect_ind_file = open(incorrect_ind_path, 'r')
    incorrect_ind = [int(x.strip()) - 1 for x in incorrect_ind_file.readlines()]

    correct_ind_file = open(correct_ind_path, 'r')
    correct_ind = [int(x.strip()) - 1 for x in correct_ind_file.readlines()]
    
    # print incorrect_ind
    
    # incorrect = []
    # correct = []

    # line_counter = 0
    # for line in csv.readlines():
    #     if line_counter in incorrect_ind:
    #         incorrect += line
    #     else:
    #         correct += line

    incorrect = picklines(csv, incorrect_ind)

    csv = open(path + "custom_features_test.csv", 'r')

    correct = picklines(csv, correct_ind)

    print len(incorrect)
    print len(correct)

    # print incorrect

    incorrect_sums = [0] * 63
    num_incorrect = 0

    for feature_vector in incorrect:
        num_incorrect += 1

        features = feature_vector.split(",")
        # print features

        index = 0
        for feature in features:
            incorrect_sums[index] += float(feature)
            index += 1

    correct_sums = [0] * 63
    num_correct = 0

    for feature_vector in correct:
        num_correct += 1

        features = feature_vector.split(",")

        index = 0
        for feature in features:
            correct_sums[index] += float(feature)
            index += 1

    # print num_correct

    incorrect_averages = [float(x) / num_incorrect for x in incorrect_sums]
    correct_averages = [float(x) / num_correct for x in correct_sums]
    averages = [(incorrect_averages[i] + correct_averages[i]) / 2.0 for i in range(len(incorrect_averages))]
    print averages
    print ""

    diffs = [incorrect_averages[i] - correct_averages[i] for i in range(len(incorrect_averages))]
    diff_fractions = [abs(diffs[i] / averages[i]) for i in range(len(incorrect_averages))]

    print diffs
    print ""
    print diff_fractions

    print ""
    print sorted(diff_fractions, reverse=True)

    print sorted(range(len(diff_fractions)), key=lambda k: diff_fractions[k], reverse=True)


if __name__ == "__main__":
    main()
