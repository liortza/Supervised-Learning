import math
import sys

VOCABULARY_SIZE = 300000  # Size of the vocabulary
UNSEEN_WORD = "unseen-word"  # Placeholder for unseen words

DEVELOP_DICT = {}  # Dictionary to hold the developed words and their counts
DEVELOP_WORDS = []  # List of words from the development file
S_SIZE = 0  # Size of the total word count

# Function to write answers to the output file
def output(output_file, answers):
    with open(output_file, 'a') as f:
        for i in range(1, len(answers) + 1):
            if i == 29:
                f.write(f"#Output{i}{answers[i - 1]}\n")
            else:
                f.write(f"#Output{i}\t{answers[i - 1]}\n")


# Function to read a file and return word counts and the list of words
def mapping_file(file):
    with open(file, 'r') as f:
        text_lines = f.readlines()

    # Split the text into words using whitespaces
    words = list()
    for i in range(2, len(text_lines), 4):  # Every 4th line starting from index 2
        words.extend(text_lines[i].split())  # Add words to list

    # Create a dictionary to count word occurrences
    word_counts = {}
    for word in words:
        word_counts[word] = word_counts.get(word, 0) + 1

    return word_counts, words


# Function implementing the Lidstone smoothing formula
def lidstone_formula(lambda_val, s, c_x, x):
    return (c_x + lambda_val) / (s + (lambda_val * x))


# Function to count the occurrence of each word in a list
def gets_words_count(words):
    word_counts = {}
    for word in words:
        word_counts[word] = word_counts.get(word, 0) + 1
    return word_counts


# Function to calculate perplexity using Lidstone smoothing
def perplexity_calc_lidstone(lambda_val, s, x, word_counts, val_words):
    probabilities = []
    for word in val_words:
        if word in word_counts:
            probabilities.append(lidstone_formula(lambda_val, s, word_counts[word], x))
        else:
            probabilities.append(lidstone_formula(lambda_val, s, 0, x))  # For unseen words
    log_sum = sum([math.log2(p) for p in probabilities if p > 0])  # Sum of log probabilities
    avg_log = -log_sum / len(val_words)  # Average log probability
    return 2 ** avg_log  # Perplexity


# Function to calculate perplexity using the held-out method
def perplexity_calc_heldout(test_words, training_words_per_count, training_words_count, held_out_words_count,
                            held_out_set_len):
    probabilities = []
    for word in test_words:
        if word in training_words_count:
            probabilities.append(
                held_out_p_calc(training_words_per_count, training_words_count, held_out_words_count, held_out_set_len,
                                word))
        else:
            probabilities.append(held_out_p_calc_unseen_words(training_words_count, held_out_words_count,
                                                              held_out_set_len))
    log_sum = sum([math.log2(p) for p in probabilities if p > 0])  # Sum of log probabilities
    avg_log = -log_sum / len(test_words)  # Average log probability
    return 2 ** avg_log  # Perplexity


# Function to compute answers based on the Lidstone smoothing model
def lidstone_answers(input_word):
    answers = []
    training_set_len = round(S_SIZE * 0.9)  # Training set length
    validation_set_len = S_SIZE - training_set_len  # Validation set length
    training_words = DEVELOP_WORDS[0: training_set_len]
    validation_words = DEVELOP_WORDS[training_set_len:]
    training_words_dict = gets_words_count(training_words)
    input_word_count_in_training_set = training_words_dict.get(input_word, 0)
    mle_input_word = input_word_count_in_training_set / training_set_len  # MLE for the input word
    unseen_word_count_in_training_set = training_words_dict.get(UNSEEN_WORD, 0)
    mle_unseen_word = unseen_word_count_in_training_set / training_set_len  # MLE for unseen words
    input_p_lidstone = lidstone_formula(lambda_val=0.10, s=training_set_len,
                                        c_x=input_word_count_in_training_set,
                                        x=VOCABULARY_SIZE)
    unseen_word_p_lidstone = lidstone_formula(lambda_val=0.10, s=training_set_len,
                                              c_x=unseen_word_count_in_training_set,
                                              x=VOCABULARY_SIZE)
    validation_set_perplexity1 = perplexity_calc_lidstone(lambda_val=0.01, s=training_set_len, x=VOCABULARY_SIZE,
                                                          word_counts=training_words_dict, val_words=validation_words)
    validation_set_perplexity2 = perplexity_calc_lidstone(lambda_val=0.1, s=training_set_len, x=VOCABULARY_SIZE,
                                                          word_counts=training_words_dict, val_words=validation_words)
    validation_set_perplexity3 = perplexity_calc_lidstone(lambda_val=1, s=training_set_len, x=VOCABULARY_SIZE,
                                                          word_counts=training_words_dict, val_words=validation_words)
    preplexity_vals = {}
    i = 0.01
    while i <= 1:  # Loop to calculate perplexity for various lambda values
        preplexity_vals[(perplexity_calc_lidstone(lambda_val=i, s=training_set_len, x=VOCABULARY_SIZE,
                                                  word_counts=training_words_dict, val_words=validation_words))] = i
        i += 0.01

    min_preplexity = min(preplexity_vals.keys())  # Minimum perplexity value
    min_lambada = preplexity_vals[min(preplexity_vals.keys())]  # Corresponding lambda value
    answers.extend([validation_set_len,
                    training_set_len,
                    len(set(training_words)),
                    input_word_count_in_training_set, mle_input_word,
                    mle_unseen_word, input_p_lidstone, unseen_word_p_lidstone, validation_set_perplexity1,
                    validation_set_perplexity2, validation_set_perplexity3, min_lambada, min_preplexity])
    # print(lidstone_check_sum_of_probabilities_equal_to_1(training_set_len, training_words_dict, unseen_word_p_lidstone))
    return answers


# Function to check if sum of probabilities equals 1 (Lidstone)
def lidstone_check_sum_of_probabilities_equal_to_1(training_set_len, training_words_dict, unseen_word_p_lidstone):
    probabilities_list = []
    for word in training_words_dict:
        word_count_in_training_set = training_words_dict[word]
        probabilities_list.append(lidstone_formula(lambda_val=0.10, s=training_set_len,
                                                   c_x=word_count_in_training_set,
                                                   x=VOCABULARY_SIZE))
    probabilities_list.append(unseen_word_p_lidstone * (VOCABULARY_SIZE - len(training_words_dict)))
    return sum(probabilities_list)  # Check if the sum of probabilities equals 1


# Function to reverse the key-value pairs in a dictionary
def values_to_keys(my_dict):
    value_to_keys = {}
    for key, value in my_dict.items():
        if value not in value_to_keys:
            value_to_keys[value] = []  # Initialize a list for the value
        value_to_keys[value].append(key)
    return value_to_keys


# Function to calculate held-out probability for a given word
def held_out_p_calc(training_words_per_count, training_words_count, held_out_words_count, held_out_set_len, word):
    word_r = training_words_count[word]
    words_r = training_words_per_count[word_r]
    tr = sum([held_out_words_count[word] for word in words_r if word in held_out_words_count])
    nr = len(words_r)
    p_held_out_input_word = tr / nr / held_out_set_len
    return p_held_out_input_word


# Function to calculate held-out probability for unseen words
def held_out_p_calc_unseen_words(training_words_count, held_out_words_count,
                                 held_out_set_len):
    tr = sum([count for word, count in held_out_words_count.items() if word not in training_words_count])
    nr = VOCABULARY_SIZE - len(training_words_count)
    p_held_out_unseen_word = tr / nr / held_out_set_len
    return p_held_out_unseen_word


# Function to check if sum of probabilities equals 1 (Held-out)
def held_out_check_sum_of_probabilities_equal_to_1(training_words_per_count, training_words, training_words_count,
                                                   held_out_words_count, held_out_set_len, unseen_word_p):
    probabilities_list = []
    set_trainig_words = set(training_words)

    for word in set_trainig_words:
        probabilities_list.append(
            held_out_p_calc(training_words_per_count, training_words_count, held_out_words_count, held_out_set_len,
                            word))
    probabilities_list.append(unseen_word_p * (VOCABULARY_SIZE - len(set_trainig_words)))
    # print(sum(probabilities_list))


# Function to compute answers based on the held-out method
def held_out_answers(input_word):
    heldout_answers = list()
    (training_set_len, held_out_set_len, training_words, held_out_words,
     training_words_count, held_out_words_count, training_words_per_count) = heldout_get_values()
    p_held_out_input_word = held_out_p_calc(training_words_per_count, training_words_count,
                                            held_out_words_count, held_out_set_len, input_word)
    p_held_out_unseen_word = 0
    if UNSEEN_WORD not in training_words_count:
        p_held_out_unseen_word = held_out_p_calc_unseen_words(training_words_count,
                                                              held_out_words_count,
                                                              held_out_set_len)
    heldout_answers.extend([training_set_len, held_out_set_len, p_held_out_input_word, p_held_out_unseen_word])
    # print(held_out_check_sum_of_probabilities_equal_to_1(training_words_per_count, training_words, training_words_count,
    #                                                      held_out_words_count, held_out_set_len,
    #                                                      p_held_out_unseen_word))
    return heldout_answers


# Function to evaluate the test set
def test_set_answer(test_file):
    test_dict, test_words = mapping_file(test_file)
    test_answers = []
    test_events = len(test_words)
    training_set_len = round(S_SIZE * 0.9)
    training_words = DEVELOP_WORDS[0: training_set_len]
    training_words_dict = gets_words_count(training_words)
    perplexity_test_lidstone = perplexity_calc_lidstone(lambda_val=0.06, s=training_set_len, x=VOCABULARY_SIZE,
                                                        word_counts=training_words_dict, val_words=test_words)
    (training_set_len, held_out_set_len, training_words, held_out_words,
     training_words_count, held_out_words_count, training_words_per_count) = heldout_get_values()
    perplexity_test_heldout = perplexity_calc_heldout(test_words,
                                                      training_words_per_count,
                                                      training_words_count,
                                                      held_out_words_count,
                                                      held_out_set_len)
    the_better_model = 'L' if perplexity_test_lidstone < perplexity_test_heldout else 'H'
    test_answers.extend([test_events, perplexity_test_lidstone, perplexity_test_heldout, the_better_model])
    return test_answers


# Function to compute the Maximum Likelihood Estimation (MLE) values for a word
def mle_value(r):
    (training_set_len, held_out_set_len, training_words, held_out_words,
     training_words_count, held_out_words_count, training_words_per_count) = heldout_get_values()
    if r:
        word_r = training_words_per_count[r][0]
        word_prob = held_out_p_calc(training_words_per_count, training_words_count, held_out_words_count,
                                    held_out_set_len, word_r)
        f_h = round(word_prob * training_set_len, 5)
    else:
        word_prob = held_out_p_calc_unseen_words(training_words_count, held_out_words_count, held_out_set_len)
        f_h = round(word_prob * training_set_len, 5)

    nrt = len(training_words_per_count[r]) if r != 0 else VOCABULARY_SIZE - len(training_words_count.keys())
    tr = sum([held_out_words_count[word] for word in training_words_per_count[r] if
              word in held_out_words_count]) if r else sum(
        [count for word, count in held_out_words_count.items() if word not in training_words_count])

    (training_set_len, training_words, training_words_dict, unseen_word_count_in_training_set,
     training_words_per_count) = lidstone_get_training_values()
    if r:
        word_r = training_words_per_count[r][0]
        input_word_count_in_training_set = training_words_dict.get(word_r, 0)
        word_prob = lidstone_formula(lambda_val=0.06, s=training_set_len,
                                     c_x=input_word_count_in_training_set,
                                     x=VOCABULARY_SIZE)
        f_lamda = round(word_prob * training_set_len, 5)
    else:
        word_prob = lidstone_formula(lambda_val=0.06, s=training_set_len,
                                     c_x=unseen_word_count_in_training_set,
                                     x=VOCABULARY_SIZE)
        f_lamda = round(word_prob * training_set_len, 5)

    return f"\n{r}\t{f_lamda}\t{f_h}\t{nrt}\t{tr}"


# Function to get values for Lidstone training
def lidstone_get_training_values():
    training_set_len = round(S_SIZE * 0.9)
    training_words = DEVELOP_WORDS[0: training_set_len]
    training_words_dict = gets_words_count(training_words)
    unseen_word_count_in_training_set = training_words_dict.get(UNSEEN_WORD, 0)
    training_words_per_count = values_to_keys(training_words_dict)

    return (training_set_len, training_words, training_words_dict,
            unseen_word_count_in_training_set, training_words_per_count)


# Function to get values for held-out training
def heldout_get_values():
    training_set_len = round(S_SIZE * 0.5)
    held_out_set_len = S_SIZE - training_set_len
    training_words = DEVELOP_WORDS[0: training_set_len]
    held_out_words = DEVELOP_WORDS[training_set_len:]
    training_words_count = gets_words_count(training_words)
    held_out_words_count = gets_words_count(held_out_words)
    training_words_per_count = values_to_keys(training_words_count)

    return (training_set_len, held_out_set_len, training_words, held_out_words,
            training_words_count, held_out_words_count, training_words_per_count)


# Main function that coordinates the flow of the program
def main():
    global DEVELOP_DIC, DEVELOP_WORDS, S_SIZE  # Global variables
    develop_file = sys.argv[1]  # Development file path
    test_file = sys.argv[2]  # Test file path
    input_word = sys.argv[3]  # Input word
    output_file = sys.argv[4]  # Output file path
    answers = []
    with open(output_file, 'w') as f:
        f.write(f"#Students\tAdva Cohen\tLior Tzahar\t323840561\t208629808\n")  # Write student information to output file
    answers.extend([develop_file, test_file, input_word, output_file, VOCABULARY_SIZE])
    DEVELOP_DICT, DEVELOP_WORDS = mapping_file(develop_file)  # Mapping the development file

    p_uniform = 1 / VOCABULARY_SIZE  # Uniform probability for unseen words
    answers.append(p_uniform)
    S_SIZE = len(DEVELOP_WORDS)  # Total number of words in the development set

    answers.append(S_SIZE)
    answers.extend(lidstone_answers(input_word))  # Add Lidstone-based answers
    answers.extend(held_out_answers(input_word))  # Add held-out model answers
    answers.extend(test_set_answer(test_file))  # Add test set answers
    mles = ""
    for i in range(10):
        mles += mle_value(i)  # Calculate MLE values for different words
    answers.extend([mles])  # Add MLE results to the answers
    output(output_file, answers)  # Write answers to the output file


if __name__ == '__main__':
    main()  # Run the main function when the script is executed
