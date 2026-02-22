import linecache

def get_test_txt(input_txt : str,
                 output_txt : str,
                 nums : int):
    with open(input_txt, 'r', encoding='utf-8') as infile, open(output_txt, 'w', encoding='utf-8') as outfile:
        for i, line in enumerate(infile, 1):
            if i > nums:
                break
            outfile.write(line)

if __name__ == "__main__":

    input_file = "data/TinyStoriesV2-GPT4-train.txt"

    output_file = "data/TinyStoriesV2-GPT4-test.txt"

    get_test_txt(input_file, output_file, 5000)
