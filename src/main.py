from src.video_processor import VideoProcessor
from src.utils import parse_arguments


def main():
    args = parse_arguments()
    video_processor = VideoProcessor(
        source_video_path=args.video,
        target_video_path=args.output,
        model_weights=args.model,
    )
    video_processor.process_video()


if __name__ == "__main__":
    main()
