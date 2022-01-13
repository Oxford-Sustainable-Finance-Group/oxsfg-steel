# oxsfg-steel
A repo for localisation and feature extraction of steel facilities in remote sensing imagery.


## Installation

## Developer Install

To setup as a developer, clone this repo and then:

```
pip install -e .[dev]
pre-commit install
```

## Useage

### Build a dataset


#### Authorization

Building a dataset requires authorization from Earth Engine and Google Maps API (to use the basemap).
To use the Earth Engine REST API, you must be an [Earth Engine user](https://earthengine.google.com/new_signup/), you must [create a GCP service account](https://support.google.com/a/answer/7378726?hl=en), download the credentials json, and [register the service account email with the Earth Engine REST API](https://signup.earthengine.google.com/#!/service_accounts).
To use the Google Maps API, the [Static Maps API](https://console.cloud.google.com/google/maps-apis/start?_ga=2.23728058.1203202661.1642099908-149915178.1642099908) must be enabled, and the key must be downloaded to a text file.
The paths to the json credentials and textfile API key must be specified in the YAML config file.

#### Builder CLI

To build a dataset, use the cli utilities in `bin/make_dataset.py`.
A dataset for Sentinel-2 and Google Basemap imagery can be made using the example YAML provided.
Call the script like so:

    python bin/make_dataset.py make-pointcentre-dataset ./data/oxsfg_steel.gpkg ./conf/build_multi_dataset.yaml ./data/test_ds/
