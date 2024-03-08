import flatbuffers
import logging

from circle_schema.circle_schema_generated import *

class CircleFBSerializer:
    '''
    Circle FlatBuffer Serializer
    '''
    def __init__(self, circle_file: str):
        with open(circle_file, 'rb') as f:
            circle_obj = Model.GetRootAs(f.read(), 0)
            self.circle_model = ModelT.InitFromObj(circle_obj)

    def inject_metadata(self, meta_name, meta_buf):
        '''Populates the metadata buffer (in bytearray) into the model file.
        Inserts meta_buf into the metadata field of Model.
        If meta_name already existed in metadata, it will be overwritten by the given buffer.
        '''
        # Prepare buffer_obj
        buffer_obj = BufferT()
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
            metadata_obj = MetadataT()
            metadata_obj.name = meta_name
            metadata_obj.buffer = len(self.circle_model.buffers) - 1
            self.circle_model.metadata.append(metadata_obj)

    def get_metadata(self, meta_name):
        if not self.circle_model.metadata:
            return None
        
        for meta in self.circle_model.metadata:
            if meta.name.decode("utf-8") == meta_name:
                return self.circle_model.buffers[meta.buffer]

    def export(self, circle_file: str):
        '''
        for meta in self.circle_model.metadata:
            buf_idx = meta.buffer

        buffer = self.circle_model.buffers[buf_idx]
        _tinfo = ctr.ModelTraining.GetRootAs(buffer.data)
        logging.debug(f"batch size is {_tinfo.BatchSize()}")
        '''
        builder = flatbuffers.Builder(0)
        builder.Finish(self.circle_model.Pack(builder))

        with open(circle_file, 'wb') as f:
            f.write(builder.Output())
        logging.info(f"save updated circle file in {circle_file}")
