import argparse

def make_args_parser():
    # create an ArgumentParser object
    parser = argparse.ArgumentParser(usage='%(prog)s [-h] [-i <input file>]')
    # fill parser with information about program arguments
    parser.add_argument('-i', '--input', action='store',
                        default=None,  metavar='',
                        help='relative path to the input file')

    # return an ArgumentParser object
    return parser.parse_args()

def print_args(args):
    print("Running with the following configuration")
    # get the __dict__ attribute of args using vars() function
    args_map = vars(args)
    for key in args_map:
        print('\t', key, '-->', args_map[key])
    # print an empty line
    print()

def main():
    # parse and print arguments
    args = make_args_parser()
    print_args(args)


if __name__ == '__main__':
    main()
