This repo is part of a project for the Computational Intelligence course at Postech (South Korea).

The main idea is to use Deep Learning techniques for subgenres classification (here techno).

Comparison between CNN-RNN and Transformers.

I used this as reference paper : Feng, M., & Feng, W. (2021). Evaluation of parallel and sequential deep learning
models for music subgenre classification. Mathematical Foundations of Computing, 4(2).

Everything was implemented by myself, Sacha Malaterre.

Two parts (still improvements to make) :
What to run and in what order.
Download and preprocessing :

- fetch/fetch_spotify_tracks.py
- fetch/download_spotify_tracks
- preprocess/create_segments
- preprocess/generate_mel_specs.py

Training Part :

- main.py

Others : 
 - Check the ```config.py```
 - You can run a dataset health check with
```bash
python -m src.utils.dataset_health --csv pathtoprocessedcsv
```
 - You can fix some issues with
```bash
python -m src.utils.clean_dataset
```

More complete Readme to come.

## License

This project is licensed under the **CC BY-NC-ND 4.0** License.  
You may not use, copy, modify, or distribute this work for commercial purposes.  
Full text: [Creative Commons License](https://creativecommons.org/licenses/by-nc-nd/4.0/)
