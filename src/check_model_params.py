import timm
from pprint import pprint

import config

model_names = timm.list_models(pretrained=True)
# model_names = timm.list_models('*eff*')
pprint(model_names)

model = "tf_efficientnet_b5"#config.MODEL_TYPE
m = timm.create_model(model, pretrained=True)

pprint(m.default_cfg)
