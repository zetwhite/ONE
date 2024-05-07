import flatbuffers
import logging

from schema import circle_schema_generated as circle
from . import train_info as tinfo

from schema import circle_traininfo_generated as ctr


class CirclePlus:
    '''
    This class worked based on object API - to rewrite flatbuffer
    '''
    train_info_name = "CIRCLE_TRAINING"

    def __init__(self, circle_file: str):
        with open(circle_file, 'rb') as f:
            circle_obj = circle.Model.GetRootAs(f.read(), 0)
            self.circle_model = circle.ModelT.InitFromObj(circle_obj)

    def __populate_metadata_buffer(self, meta_name, meta_buf):
        # Prepare buffer_obj
        buffer_obj = circle.BufferT()
        buffer_obj.data = meta_buf

        is_populated = False
        if not self.circle_model.metadata:
            self.circle_model.metadata = []
        else:
            # Check if metadata has already been populated.
            for meta in self.circle_model.metadata:
                if meta.name.decode("utf-8") == meta_name:
                    is_populated = True
                    self.circle_model.buffers[meta.buffer] = buffer_obj

        if not is_populated:
            if not self.circle_model.buffers:
                self.circle_model.buffers = []
            self.circle_model.buffers.append(buffer_obj)
            # Creates a new metadata field.
            metadata_obj = circle.MetadataT()
            metadata_obj.name = meta_name
            metadata_obj.buffer = len(self.circle_model.buffers) - 1
            self.circle_model.metadata.append(metadata_obj)

    def inject_tinfo_as_metadata(self, train_info):
        tinfo_name = self.train_info_name
        tinfo_buff = train_info.get_buff()
        self.__populate_metadata_buffer(tinfo_name, tinfo_buff)
        logging.info(f"sucessfully inject training parameter")

    def export(self, circle_file: str):
        builder = flatbuffers.Builder(0)
        builder.Finish(self.circle_model.Pack(builder))

        with open(circle_file, 'wb') as f:
            f.write(builder.Output())
        logging.info(f"save updated circle file in {circle_file}")
