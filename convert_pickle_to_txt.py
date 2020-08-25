import pickle
import os

def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Show and convert a python pickle file to txt file")    
    parser.add_argument('pickle_file',  type=str, default = None, help = 'The pickle file path')

    return parser.parse_args()


def main(args):
    # base_filename = os.path.splitext(args.pickle_file)[0]
    # path_name = os.path.dirname(args.pickle_file)
    # output_filename = os.path.join(path_name, base_filename + '.txt')
    text_file = open(args.pickle_file + ".txt", "w")

    dbfile = open(args.pickle_file, 'rb')
    db = pickle.load(dbfile)
    for keys in db:
        print(keys, '=>', db[keys])
        text_file.write("{} => {}\n".format(keys, db[keys]))
    dbfile.close()
    text_file.close()
    

if __name__ == "__main__":
    args = parse_args()
    main(args)
  