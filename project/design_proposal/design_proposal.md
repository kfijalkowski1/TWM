# Segmentacja semantyczna obrazów w warunkach mgły

Celem projektu jest analiza wpływu obecności mgły na jakość modeli segmentacji semantycznej obrazów oraz ocena potencjalnych metod poprawy ich skuteczności 

## Istniejące rozwiązania

### Inne prace na podobne tematy:
Istnieje wiele prac dotykających podobne zagadnienia, większość z nich jest bardzo zaawansowanymi rozwiązaniami prezentującymi wspaniałe wyniki, aczkolwiek w ramach naszego podejścia chcieliśmy wykorzystać istniejące architektury w celu sprawdzenia ich skuteczności
- [Learning Fog-invariant Features for Foggy Scene Segmentation](https://arxiv.org/abs/2204.01587) - bardzo dobre wyniki, aczkolwiek złożona implementacja
- [FogAdapt: Self-Supervised Domain Adaptation for Semantic Segmentation of Foggy Images](https://arxiv.org/abs/2201.02588)


### Popularne sposoby usuwania mgły
- [Dark Channel Prior](https://www.sciencedirect.com/topics/computer-science/dark-channel-prior) -- Bardzo skuteczna przy umiarkowanej mgle, ale wolna
 - [CLAHE](https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html) - Tak na prawdę zwiększa kontrast lokalny, szybka i prosta, ale nie usuwa mgły i może pogorszyć obraz



## Zbiory danych 

### CityScapes

- Standardowy zbiór danych dla zadania segmentacji semantycznej
- Zdjęcia RGB obszarów miejskich z maskami dla 30 klas (możliwe zgrupowanie w 8 bardziej ogólnych)
- 5000 zdjęć z adnotacjami wysokiej jakości, 20000 zgrubnie poetykietowanych
- https://www.cityscapes-dataset.com/
- https://paperswithcode.com/dataset/cityscapes

### Foggy CityScapes

- Rozszerzenie zbioru CityScapes o syntetyczną mgłę wygenerowaną na podstawie map głębii
- 3 poziomy gęstości mgły równoważne widoczności 600m, 300m i 150m
- Adnotacje te same, co w oryginalnym CityScapes

- https://arxiv.org/abs/1708.07819
- http://people.ee.ethz.ch/~csakarid/SFSU_synthetic/
- https://paperswithcode.com/dataset/foggy-cityscapes

## Architektury modeli

- DeepLabV3
  - Klasyczna i zrozumiała architektura
  - Wysoki wynik dla CityScapes
  - Wersja pre-trenowana na MS COCO do pobrania, można zrobić finetuning
    - https://pytorch.org/hub/pytorch_vision_deeplabv3_resnet101/
  - Ewentualnie nieco mocniejsza odmiana DeepLabV3+ (enkoder-dekoder), dostępna wersja pre-trenowana na CityScapes:
    - https://github.com/VainF/DeepLabV3Plus-Pytorch
  - Backbone - do ustalenia
- DehazeFormer
  - Przetwarzanie image-to-image, w tym wypadku usuwanie mgły ze zdjęć
  - SOTA dla zbioru danych RESIDE-6K
  - Dość dobra implementacja - powinno się dać podpiąć własny dataset
  - https://paperswithcode.com/paper/vision-transformers-for-single-image-dehazing
  - https://github.com/IDKiro/DehazeFormer

## Planowane eksperymenty

Porównanie jakości segmentacji dla scenariuszy:

- "Zero shot" - model trenowany na CS (CityScapes), ewaluacja bezpośrednio na FCS (Foggy CityScapes)
- Model trenowany na CS, usuwanie mgły HazeFormerem bez dostrajania i ewaluacja na FCS
- Model trenowany na CS, usuwanie mgły HazeFormerem dostrojonym na podzbiorze FCS i ewaluacja na jego pozostałej części
- Model trenowany na CS dostrojony na FCS

Do rozważenia:
- Generalizacja modelu do zdjęć zwykłych i zamglonych przez regularyzację ukrytych reprezentacji
- Fine-tuning całej sieci vs samego dekodera - innymi słowy, czy zdjęcia zamglone wymagają innej reprezentacji, czy reprezentacja jest w porządku, ale zmienia się sposób jej dekodowania

## Implementacja

- PyTorch / PyTorch Lightning
- Konfiguracja eksperymentów - Hydra
- Śledzenie eksperymentów - Weights and Biases
- Strojenie hiperparametrów - Optuna / Weights and Biases Sweeps
- Implementacje modeli:
  - DeepLabV3(+): 
    - https://pytorch.org/hub/pytorch_vision_deeplabv3_resnet101/
    - https://github.com/VainF/DeepLabV3Plus-Pytorch
  - DehazeFormer:
    - https://github.com/IDKiro/DehazeFormer
