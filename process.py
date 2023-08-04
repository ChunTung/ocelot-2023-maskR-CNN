from util import gcio
from util.constants import (
    GC_CELL_FPATH, 
    GC_TISSUE_FPATH, 
    GC_METADATA_FPATH,
    GC_DETECTION_OUTPUT_PATH
)
from user.inference import Model
import pdb
import time


def process():
    """Process a test patches. This involves iterating over samples,
    inferring and write the cell predictions
    """
    # Initialize the data loader
    # pdb.set_trace()
    loader = gcio.DataLoader(GC_CELL_FPATH, GC_TISSUE_FPATH)

    # Cell detection writer
    writer = gcio.DetectionWriter(GC_DETECTION_OUTPUT_PATH)
    
    # Loading metadata
    meta_dataset = gcio.read_json(GC_METADATA_FPATH)

    # Instantiate the inferring model
    model = Model(meta_dataset)
    # print(model)
    model_weight = './mask_rcnn_nucleus_0095.h5'

    # NOTE: Batch size is 1
    for cell_patch, tissue_patch, pair_id in loader:
        print(cell_patch.shape)
        print(f"Processing sample pair {pair_id}")
        # pdb.set_trace()
        # Cell-tissue patch pair inference
        cell_classification = model(cell_patch, tissue_patch, pair_id,model_weight)
        
        # Updating predictions
        # pdb.set_trace()
        writer.add_points(cell_classification, pair_id)

    # Export the prediction into a json file
    writer.save()


if __name__ == "__main__":
    # start = time.time()
    process()
    # end = time.time()
    # print(end-start)