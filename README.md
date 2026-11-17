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
