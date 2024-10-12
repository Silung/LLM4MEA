
from datasets import dataset_factory
from utils import *

set_template(args)

org_dataset = dataset_factory(args)
org_dataset = org_dataset.load_dataset()
org_data = org_dataset['train']
org_data = [len(v) for v in org_data.values()]
print(np.array(org_data).mean())