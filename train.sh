#!/usr/bin/env bash

for dir in data/youtube_ids/*.txt; do 

	while read -r line; do 

		if [[ $(echo $line | wc -c) == 12 ]]; then

			data/download.sh "$(echo -n $line)"

		fi

	done < $dir

done	
