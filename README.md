# Address Parsing Tests

Train a Modern BERT model on OSM data to parse free-text addresses into structured data. Inspired by [libpostal](https://github.com/openvenues/libpostal).
E.g.  -> {"house_number": "241", "street": "17th Cross Road, 9th Main Road"}

```
addr: No. 241, 17th Cross Road, 9th Main Road, Sector 1 HSR Layout, Bengaluru, 560102
  city          Bengaluru
  house_number  241
  postcode      560102
  road          17th Cross Road 9th Main Road
  suburb        Sector 1 HSR Layout
```

# Why

`libpostal`'s models (both the 'original' and the Senzing models) don't handle Indian addresses very well. This boils down to three problems

1. Indian address data is lacking in OSM (and other open-data sources) - many neighbourhoods, house numbers and streets are not on OSM. This situation has improved over the years.
2. Indian address _formats_ are underrepresented in the OpenCage address formats.
3. A lot of _incorrect_ data is in the Indian addresses that are entered. e.g. `addr:street` often includes the full address, `addr:housenumber` contains "No", "SyNO" and other prefixes, and `name` often includes mulitiple names.

1 can be resolved by adding more data to OSM - this takes time, as there are few data sources to use for imports, which means it relies on ground surveys.
2 can be fixed by a trivial PR to the OpenCage address formats (and must be done anyway). 3 must also be fixed by reviewing data on OSM using data quality tools like osmose.

Since it will take time to fix 1 and 3, the idea is to train a model by cleaning the data with basic rules - e.g. don't include any `addr:street` values, drop prefixes from `addr:housenumber`, etc.

# Running

1. Fetch data from geofabrik - e.g. `southern-zone-latest.osm.pbf` into `datasets/`
2. Generate the dataset
```bash
cd osm_address_extractor

podman build -t osm2parquet . && cd ..

podman run --security-opt label=disable --rm \
  -v osm_dub:/root/.dub -v $PWD:/geo \
  -e LD_LIBRARY_PATH=/geo/thirdparty/shared --entrypoint /bin/bash osm2parquet \
  -c "cd /geo/osm_address_extractor && dub build --compiler=ldc2 -b release && \
      sh run.sh -i /geo/datasets/southern-zone-latest.osm.pbf \
        -o data/sz-buckets \
        --train data/sz-train.jsonl --test data/sz-test.jsonl"
```
3. Train the model
```bash
python3 -m venv .venv
.venv/bin/pip install torch transformers

.venv/bin/python train.py --train osm_address_extractor/data/sz-train.jsonl \
    --test osm_address_extractor/data/sz-test.jsonl --epochs 2 --out model
```
4. Evaluate and predict
```bash
.venv/bin/python eval.py --model model \
    --test osm_address_extractor/data/sz-test.jsonl

# source one address per line; use --input <file> to read from a file
.venv/bin/python predict.py --model model
```

Evaluation is against 100 hand-labeled samples from [BBMP Trade License data](https://github.com/Vonter/bbmp-trade-license). More can be added as the model improves.
