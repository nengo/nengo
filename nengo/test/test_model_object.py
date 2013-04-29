
import test_model
from nengo import model_object

# TODO: import other unit tests that use model.Model
#       and swap in model_object.Model

class TestModelObject(test_model.TestModel):
    Model = model_object.Model


