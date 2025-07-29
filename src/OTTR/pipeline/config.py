import os
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Union

import yaml

def merge_namespaces(a: SimpleNamespace, b: SimpleNamespace) -> None:
	for k, v in vars(b).items():
		if isinstance(v, SimpleNamespace):
			merge_namespaces(a.__dict__[k], v)
		else:
			a.__dict__[k] = v


class Config(SimpleNamespace):
	def to_string(self, offset: int = 0) -> str:
		s = ""
		for k, v in vars(self).items():
			s += "".join([" "] * offset)
			if isinstance(v, SimpleNamespace):
				s += f"{k}:\n{v.to_string(offset + 2)}"
			else:
				s += f"{k}: {v}\n"
		return s

	def merge_with(self, path: str) -> None:
		if not os.path.exists(path):
			raise IOError(f"Config path {path} not found.")

		with open(path) as f:
			defs = yaml.load(f, Loader=yaml.Loader)

		if "paths" in defs:
			merge_namespaces(self.paths, SimpleNamespace(**defs["paths"]))
		if "params" in defs:
			merge_namespaces(self.params, SimpleNamespace(**defs["params"]))
		if "steps" in defs:
			self.steps = defs["steps"]
		if "execution" in defs:
			merge_namespaces(self.execution, SimpleNamespace(**defs["execution"]))


# def load_config(subset_obj: Union[Optional[PunchcardSubset], Optional[PunchcardView]] = None) -> Config:
def load_config(build:str = None) -> Config:
	config = Config(**{
		"paths": Config(**{
            "build": "",
            "raw_data": "",
			"plot_dir": ""
		}),
		"params": Config(**{
            "binning": 1,
			"FPS": 24,
			"segmentation_method": 'watershed', ## Or cellpose
			"cell_diameter": 12, ## Expected cell sizes
			"overlap": .1, ## .05
			"Tissue": "Cortex",
			"Virus": "H2B-YFP",
			"SampleID": "",
			"size_lim": [25,4000],
			"min_frames": 10
		}),
		"steps": (),
		"execution": Config(**{
			# "n_cpus": available_cpu_count(),
			"n_cpus": 4,
			"n_gpus": 1,
			"memory": 8
		})
	})

	if build:
		config.paths.build = build
	# Home directory
	f = os.path.join(os.path.abspath(str(Path.home())), ".embryoscope")
	if os.path.exists(f):
		config.merge_with(f)
	# Set build folder
	if config.paths.build == "" or config.paths.build is None:
		config.paths.build = os.path.abspath(os.path.curdir)
	# Build folder
	f = os.path.join(config.paths.build, "config.yaml")
	if os.path.exists(f):
		config.merge_with(f)

	return config