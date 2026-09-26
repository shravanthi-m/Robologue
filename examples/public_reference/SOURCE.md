# Public IndustReal component reference

Attribution: Tim J. Schoonbeek, Tim Houben, Hans Onvlee, Peter H. N. de With, and Fons van der Sommen, *IndustReal Dataset of Egocentric Videos for Procedure Understanding*, version 2. The construction model originates from the STEMFIE Project by Paulo Kiefe (https://stemfie.org/sps-000001).

Dataset: https://data.4tu.nl/datasets/b008dd74-020d-4ea4-a8ba-7bb60769d224
Version DOI: https://doi.org/10.4121/b008dd74-020d-4ea4-a8ba-7bb60769d224.v2
Exact public geometry archive: https://data.4tu.nl/file/b008dd74-020d-4ea4-a8ba-7bb60769d224/4b6641cf-0034-4875-8ebb-73e35e4a5b91
Archive filename: `part_geometries.zip`; published size: 17,338,432 bytes.
Archive MD5, verified after download: `20a25a8eaa2ad605f269d30fa40275a7`.
Retrieved 2026-09-26 through BrowserOS Neo without login.

The official dataset metadata and reference repository declare Apache-2.0. `LICENSE` is the complete license text copied unchanged from https://github.com/TimSchoonbeek/IndustReal/blob/main/license.txt (the local clone at retrieval).

## Included assets and modifications

`component_key.jpg` is a modified rendition of page 1 of `part_geometries/Overview of states.pdf`. Page 1's public labeled CAD component diagram was rendered and cropped. No individual recording annotations or binary state tables are included. The crop preserves the original labels, CAD views, and arrows; no generated imagery was added.

Reproduction command, using Poppler:

```sh
pdftoppm -f 1 -l 1 -singlefile -scale-to 1800 -x 12 -y 740 -W 1370 -H 735 -jpeg -jpegopt quality=95 'part_geometries/Overview of states.pdf' component_key
```

The unrotated page is US Letter, rendered with a 1800-pixel long edge. Crop coordinates are x=12, y=740, width=1370, height=735 pixels from the rendered page's top left. JPEG dimensions are 1370 by 735. Rendering used installed system fonts; a different Poppler/font setup can change bytes. The distributed JPEG hash is recorded in `public_components.json`.

`public_components.json` is a new, manually authored derivative. It maps component-0 through component-10 to bounded descriptions grounded in the page 1 labeled diagram. It includes exact source provenance and SHA-256 hashes for the source PDF and packaged JPEG. Its relative image filename is `component_key.jpg`.

Descriptions identify component geometry and attachment positions. They contain no frame labels, recording references, per-recording state values, or binary state codes. Observed assembly state still requires independent RGB evidence. This key supplies public semantic grounding for front/rear component identities.

The archive, source PDF, complete state tables, CAD models, and recordings are deliberately outside this package. Download the exact public archive above if the original document is needed.
