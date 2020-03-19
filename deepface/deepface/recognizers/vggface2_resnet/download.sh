#!/bin/bash

echo "[download] model graph : VGGFace-Resnet"
DIR="$(cd "$(dirname "$0")" && pwd)"

extract_download_url() {
        if ! [ -x "$(command -v wget)" ]; then
          echo 'Error: wget is not installed.' >&2
          exit 1
        fi
        url=$( wget -q -O - $1 |  grep -o 'http*://download[^"]*' | tail -n 1 )
        echo "$url"

}

extract_filename() {
        echo "$DIR/${1##*/}"
}

download_mediafire() {
        if ! [ -x "$(command -v curl)" ]; then
          echo 'Error: curl is not installed.' >&2
          exit 1
        fi
        curl -L -o $( extract_filename $1 ) -C - $( extract_download_url $1 )
}

$( download_mediafire http://www.mediafire.com/file/datqeencsrqua22/rcmalli_vggface_tf_resnet50.h5 )
$( download_mediafire http://www.mediafire.com/file/sc2fo5xcktwbqkd/rcmalli_vggface_labels_v2.npy )