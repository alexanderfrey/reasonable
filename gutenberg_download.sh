#!/bin/bash

# Default values
start_index=1
end_index=100
download_dir="gutenberg_books"
delay=2  # Delay between downloads in seconds

# Parse command line arguments
while getopts "s:e:d:t:" opt; do
    case $opt in
        s) start_index=$OPTARG ;;
        e) end_index=$OPTARG ;;
        d) download_dir=$OPTARG ;;
        t) delay=$OPTARG ;;
        \?) echo "Invalid option: -$OPTARG" >&2; exit 1 ;;
    esac
done

# Create download directory if it doesn't exist
mkdir -p "$download_dir"

# Download books
for i in $(seq $start_index $end_index); do
    url="https://www.gutenberg.org/cache/epub/$i/pg$i.txt"
    output_file="$download_dir/book_$i.txt"
    
    echo "Downloading book $i..."
    if curl -f -s "$url" -o "$output_file"; then
        echo "Successfully downloaded book $i"
    else
        echo "Failed to download book $i"
        rm -f "$output_file"  # Remove empty/partial file
    fi
    
    # Respect server by waiting between downloads
    sleep $delay
done

echo "Download complete. Books saved in $download_dir/"