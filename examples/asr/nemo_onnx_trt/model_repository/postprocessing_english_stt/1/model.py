import numpy as np
import torch
import triton_python_backend_utils as pb_utils
import nemo.collections.asr as nemo_asr

class TritonPythonModel:
    def initialize(self, args):
        asr_model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.from_pretrained(
            model_name="nvidia/stt_en_fastconformer_ctc_large",
            map_location='cpu'
        )
        asr_model.ctc_decoding.cfg.strategy = 'greedy_batch'
        self.ctc_decoder = asr_model.decoding.ctc_decoder_predictions_tensor
        del asr_model

    def execute(self, requests):
        responses = []

        for request in requests:
            asr_output = pb_utils.get_input_tensor_by_name(request, "asr_output").as_numpy()
            ctc_output_tensor = torch.from_numpy(asr_output)
            decoded_texts = self.ctc_decoder(ctc_output_tensor, decoder_lengths=None)
            decoded_texts = [i.text for i in decoded_texts]

            decoded_texts_numpy = np.array(decoded_texts, dtype=np.object)

            output_tensor = pb_utils.Tensor("decoded_texts", decoded_texts_numpy)

            inference_response = pb_utils.InferenceResponse(output_tensors=[output_tensor])
            responses.append(inference_response)

        return responses

    def finalize(self):
        pass