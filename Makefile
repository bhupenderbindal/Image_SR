.PHONY: clean data lint requirements sync_data_to_s3 sync_data_from_s3

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
BUCKET = [OPTIONAL] your-bucket-for-syncing-data (do not include 's3://')
PROFILE = default
PROJECT_NAME = ma_imagesr
PYTHON_INTERPRETER = python3

ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
endif

#################################################################################
# COMMANDS                                                                      #
#################################################################################


EXP = test
#channel type y or rgb
CT = rgb
# batch size
BS = 4
# data type div2k or fdata
DT = div2k
# num of epochs
EP = 1 
LR = 0.0001
# setnum for fdata - 1 or 2 or 3 (test data)
SETNUM = 1
# loss function for exmaple based - cobi, l2, l1
LOSSTYPE=cobi


zip_fdata:
	cd ./data/processed/ ;\
	echo $(PWD); \
	./fdata_zip.sh


create_samplewise_dataset:
	@echo "current working directory:" $(PWD); 
	@echo "========== creating paired dataset ======="; \
	python -m src.data.paired_data_generator_withalignment_samplewise  --n 9 --set_num $(SETNUM)


srcnn:
	@echo "========== training and testing srcnn ======="; \
	echo $(PWD); \
	python -m src.srcnn.main --model srcnn --batchSize $(BS) --lr $(LR) --upscale_factor 2 --nEpochs $(EP) --datatype $(DT) --channeltype $(CT) --expname $(EXP) --losstype $(LOSSTYPE) --set_num $(SETNUM); \


dbpn:
	@echo "========== training and testing dbpn ======="; \
	echo $(PWD); \
	python -m src.srcnn.main --model dbpn --batchSize $(BS) --lr $(LR) --upscale_factor 2 --nEpochs $(EP) --datatype $(DT) --channeltype $(CT) --expname $(EXP) --losstype $(LOSSTYPE) --set_num $(SETNUM); \
	

kernelgandiv2k:
	@echo "========== training and testing KernelGAN + ZSSR ======="; \
	echo $(PWD); \
	python -m src.KernelGAN-master_v1.train --SR --real --paired-data-dir ./data/raw/Set5
	python -m src.KernelGAN-master_v1.train --SR --real --paired-data-dir ./data/raw/div2k_data/val/
	python -m src.KernelGAN-master_v1.train --SR --real --paired-data-dir ./data/raw/div2k_data/test/

kernelgandiv2krk:
	@echo "========== training and testing KernelGAN + ZSSR ======="; \
	echo $(PWD); \
	python -m src.KernelGAN-master_v1.train --SR --real --paired-data-dir ./data/raw/div2krk/val/

kernelganset1:
	@echo "========== training and testing KernelGAN + ZSSR ======="; \
	echo $(PWD); \
	python -m src.KernelGAN-master_v1.train --SR --real --paired-data-dir ./data/processed/set_1/img_pairs_train_val_test/val/

kernelganset2:
	@echo "========== training and testing KernelGAN + ZSSR ======="; \
	echo $(PWD); \
	python -m src.KernelGAN-master_v1.train --SR --real --paired-data-dir ./data/processed/set_2/img_pairs_train_val_test/val/
	
kernelganset3:
	@echo "========== training and testing KernelGAN + ZSSR ======="; \
	echo $(PWD); \
	python -m src.KernelGAN-master_v1.train --SR --real --paired-data-dir ./data/processed/set_3/img_pairs_train_val_test/val/


# install panel package	
vizapp:
	panel serve src/visualization/viztool/vizapp2.py  --autoreload --port 5008

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint  
lint:
	ruff check ./src/ --ignore=F841,E741

## Set up python interpreter environment
create_environment:
ifeq (True,$(HAS_CONDA))
		@echo ">>> Detected conda, creating conda environment."
ifeq (3,$(findstring 3,$(PYTHON_INTERPRETER)))
	conda create --name $(PROJECT_NAME) python=3
else
	conda create --name $(PROJECT_NAME) python=2.7
endif
		@echo ">>> New conda env created. Activate with:\nsource activate $(PROJECT_NAME)"
else
	$(PYTHON_INTERPRETER) -m pip install -q virtualenv virtualenvwrapper
	@echo ">>> Installing virtualenvwrapper if not already installed.\nMake sure the following lines are in shell startup file\n\
	export WORKON_HOME=$$HOME/.virtualenvs\nexport PROJECT_HOME=$$HOME/Devel\nsource /usr/local/bin/virtualenvwrapper.sh\n"
	@bash -c "source `which virtualenvwrapper.sh`;mkvirtualenv $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER)"
	@echo ">>> New virtualenv created. Activate with:\nworkon $(PROJECT_NAME)"
endif

## Test python environment is setup correctly
test_environment:
	$(PYTHON_INTERPRETER) test_environment.py

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################



#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
