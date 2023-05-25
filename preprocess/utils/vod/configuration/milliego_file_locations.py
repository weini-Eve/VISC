import os


class MilliegoLocations:
    """
    This class contains the information regarding the locations of data for the dataset.
    """
    def __init__(self, root_dir: str, sub_dir: str, output_dir: str = None, frame_set_path: str = None, pred_dir: str = None):
        """
Constructor which based on a few parameters defines the locations of possible data.
        :param root_dir: The root directory of the dataset.
        :param sub_dir: The subdirectory under the root.
        :param output_dir: Optional parameter of the location where output such as pictures should be generated.
        :param sub_dir: Optional parameter of the text file of which output should be generated.
        :param pred_dir: Optional parameter of the locations of the prediction labels.
        """

        # Input parameters
        self.root_dir: str = root_dir
        self.sub_dir: str = sub_dir
        self.output_dir: str = output_dir
        self.frame_set_path: str = frame_set_path
        self.pred_dir: str = pred_dir

        # Automatically defined variables. The location of sub-folders can be customized here.
        # Current definitions are based on the recommended locations.
        self.camera_dir = os.path.join(self.root_dir, self.sub_dir, 'rgb')
        self.radar_dir = os.path.join(self.root_dir, self.sub_dir, 'mmwave_middle_pcl')
        self.imu_dir = os.path.join(self.root_dir, self.sub_dir)
        self.odo_dir = os.path.join(self.root_dir, self.sub_dir)