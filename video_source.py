import cv2
import logging

logger = logging.getLogger(__name__)

class VideoSource:
    def __init__(self, source):
        self.source = source
        self._cap = cv2.VideoCapture(source)
        if not self._cap.isOpened():
            raise FileNotFoundError(f"Cannot open video source: {source}")
        self.width  = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps    = self._cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info("Opened '%s'  %dx%d  %.1f fps  %d frames",
                    source, self.width, self.height, self.fps, self.total_frames)

    def __iter__(self):
        return self

    def __next__(self):
        ret, frame = self._cap.read()
        if not ret:
            raise StopIteration
        return frame

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.release()

    def release(self):
        self._cap.release()
        logger.debug("VideoCapture released.")
