# Third-party notices

## LADA

This add-on adapts LADA revision `20cb34a20a83c72c87a991d2c949032c70085b16`.

- Project: https://github.com/ladaapp/lada
- License: GNU Affero General Public License v3.0 only
- Matching source: the `lada/` directory in this release's `source.tar.zst`

## LADA first-party model weights

The model bundle contains weights published by `ladaapp/lada` at Hugging Face revision `bcf461d46d9a98981fc64b815df5178f42215cdf`. LADA documents its first-party model weights as AGPL-3.0. Exact filenames, source URLs, sizes, and SHA-256 digests are recorded in `manifests/models.json`.

The optional third-party DeepMosaics checkpoint is deliberately excluded.

## Python and accelerator dependencies

Release bundles contain the dependency closure selected by the pinned upstream `uv.lock`, including PyTorch, torchvision, PyAV, Ultralytics, OpenCV, MMEngine, and their transitive dependencies. Their individual license files and metadata must be retained from the installed distributions in every binary bundle. `packaging/build-bundles.sh` and `packaging/lada.uv.lock` are part of the Corresponding Source used to reproduce the bundle.
