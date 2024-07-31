#!/bin/bash
# removing previous zip otherwise both are combine

rm inferencedata.zip


zip -r -o inferencedata.zip ./all_data/Images_set3/ 
#zip -r -o div2k_data.zip ./div2k_data/ 

