from model import train_and_evaluate
from utilities import build_argparser, answer_from_passage


def main():
    parser = build_argparser()
    args = parser.parse_args()

    if args.mode == "train":
        train_and_evaluate(args)
    elif args.mode == "infer":
        answer_from_passage(args)
    else:
        raise ValueError("Unknown mode.")


if __name__ == "__main__":
    main()
