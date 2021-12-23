import csv
from typing import Callable, Tuple

import matplotlib.pyplot as plt
import numpy as np

INVALID_VALUE = -1
VALID_YEARS = [2012, 2013, 2015, 2016, 2017, 2018, 2019, 2020]
VALID_DAYS = range(200, 282)  #range(0, 365)  #

Pixel = list[float]
Band = list[float]

RED = 0
NIR = 1
BLUE = 2
GREEN = 3


class Sample:
    acquisition_day: int
    acquisition_year: int
    product: str
    bands: list[Band]
    pixels: list[Pixel]

    def __init__(self, acquisition_day: int, acquisition_year: int, product: str, bands: list[Band]):
        self.acquisition_day = acquisition_day
        self.acquisition_year = acquisition_year
        self.product = product
        self.bands = bands

        self.pixels = []

        for i in range(len(bands[0])):
            pixel = []

            for digital_values in self.bands:
                pixel.append(digital_values[i])

            self.pixels.append(pixel)

    def valid(self) -> bool:
        for digital_values in self.bands:
            if any(digital_value == INVALID_VALUE for digital_value in digital_values):
                return False

        return self.acquisition_year in VALID_YEARS and self.acquisition_day in VALID_DAYS

    def __str__(self) -> str:
        desc = f"{self.acquisition_day}\n" \
            f"{self.acquisition_year}\n" \
            f"{self.product}\n"

        for i, digital_values in enumerate(self.bands):
            desc += f"{i}: {digital_values}\n\n"

        return desc


def read_samples_from_file(csv_file_path: str, number_of_bands: int, pixels_of_interest: list[int], pixel_data_offset: int, rows_to_skip: int, parse_acquisition_day: Callable[[list[str]], int], parse_acquisition_year: Callable[[list[str]], int], parse_product: Callable[[list[str]], str]) -> list[Sample]:
    samples = []

    with open(csv_file_path, newline='') as csvfile:
        csv_reader = csv.reader(csvfile)

        for _ in range(rows_to_skip + 1):
            row = next(csv_reader)

        keep_parsing = True

        while keep_parsing:
            bands: list[Band] = []

            acquisition_day = parse_acquisition_day(row)
            acquisition_year = parse_acquisition_year(row)
            product = parse_product(row)

            for _ in range(number_of_bands):
                digital_values: Band = []

                for pixel in pixels_of_interest:
                    cell = row[pixel + pixel_data_offset]

                    if cell in ['F', 'NA']:
                        digital_values.append(INVALID_VALUE)
                    else:
                        digital_values.append(float(cell))

                bands.append(digital_values)

                try:
                    row = next(csv_reader)
                except StopIteration:
                    keep_parsing = False

            sample = Sample(acquisition_day, acquisition_year,
                            product, bands.copy())

            if sample.valid():
                samples.append(sample)

    return samples


def sample_avg(sample: Sample, callable: Callable[[Pixel], float], params, complement: bool = False):
    if complement:
        return sum(1 - callable(pixel, **params) for pixel in sample.pixels) / len(sample.pixels)
    else:
        return sum(callable(pixel, **params) for pixel in sample.pixels) / len(sample.pixels)


def ndvi(pixel: Pixel) -> float:
    return (pixel[NIR] - pixel[RED])/(pixel[NIR] + pixel[RED])


def wdrvi(pixel: Pixel, alpha: float = 0.25) -> float:
    return (alpha*pixel[NIR] - pixel[RED])/(alpha*pixel[NIR] + pixel[RED])


def evi3(pixel: Pixel) -> float:
    G = 2.5
    C1 = 6
    C2 = 7.5
    L = 1
    return G * (pixel[NIR] - pixel[RED])/(pixel[NIR] + C1*pixel[RED] + C2*pixel[BLUE] + L)


def evi2(pixel: Pixel) -> float:
    G = 2.5
    C1 = 2.4
    L = 1
    return G * (pixel[NIR] - pixel[RED])/(pixel[NIR] + C1*pixel[RED] + L)


def red_ratio(pixel: Pixel) -> float:
    return pixel[RED]/(pixel[RED] + pixel[BLUE] + pixel[GREEN])


def green_ratio(pixel: Pixel) -> float:
    return pixel[GREEN]/(pixel[RED] + pixel[BLUE] + pixel[GREEN])


def tasseled_cap_greenness(pixel: Pixel) -> float:
    # Coef from https://gis.stackexchange.com/questions/351077/how-to-calculate-and-export-tasseled-caps-from-modis-collection-in-gee
    return -0.4064*pixel[0] + 0.5129*pixel[1] + -0.2744*pixel[2] + -0.2893*pixel[3] + 0.4882*pixel[4] + -0.0036*pixel[5] + -0.4169*pixel[6]


def polynomial_to_str(poly_coef: list[float], label_x: str, label_y: str) -> str:
    poly_coef = [round(coef, 5) for coef in poly_coef]

    string = f'{label_y} = '

    for index, coef in enumerate(poly_coef[:-2]):
        string += f'{coef}({label_x})^{len(poly_coef) - index - 1} '

    string += f'{poly_coef[-2]}({label_x}) + {poly_coef[-1]}'

    return string


def r_squared(data_x, data_y) -> float:
    correlation_matrix = np.corrcoef(data_x, data_y, rowvar=True)
    return round(correlation_matrix[0][1]**2, 5)


def correlation_graph(label_x: str, data_x: list[float], label_y: str, data_y: list[float]) -> None:
    _, ax = plt.subplots()

    handles = []
    labels = []

    ax.scatter(data_x, data_y)

    reg_coef = np.polyfit(data_x, data_y, 1)
    reg_fn = np.poly1d(reg_coef)
    estimated_y = [reg_fn(x) for x in data_x]
    handles.append(plt.plot(data_x, estimated_y, 'k')[0])
    labels.append(f'Line of best fit\n{polynomial_to_str(reg_coef, label_x, label_y)}\nr^2 = {r_squared(data_y, estimated_y)}')

    print(polynomial_to_str(reg_coef, label_x, label_y))

    ax.set_title(f'Correlation between {label_x} and {label_y}')
    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)
    ax.legend(handles, labels)


def estimation_graph(label_x: str, data_x: list[float], label_y1: str, data_y1: list[float], label_y2: str, data_y2: list[float]) -> None:
    _, ax = plt.subplots(figsize=(12,10))

    print(len(data_y2))

    handles = []
    labels = []

    handles.append(plt.scatter(data_x, data_y1, c='b'))
    labels.append(label_y1)

    y2_y1_reg_coef = np.polyfit(data_y1, data_y2, 1)
    y2_y1_reg_fn = np.poly1d(y2_y1_reg_coef)

    estimated_y2 = [y2_y1_reg_fn(y1) for y1 in data_y1]
    handles.append(plt.scatter(data_x, estimated_y2, c='r'))
    labels.append(f'Estimated {polynomial_to_str(y2_y1_reg_coef, label_y1, label_y2)}\nr^2 = {r_squared(data_y2, estimated_y2)}')

    handles.append(plt.scatter(data_x, data_y2, c='g'))
    labels.append(label_y2)

    ax.set_title(f'{VALID_YEARS[0]}-{VALID_YEARS[-1]} Estimate of {label_y2} using {label_y1}')
    ax.set_xlabel(label_x)
    ax.legend(handles, labels)


def standard_graph(title: str, label_x: str, datasets: dict[str, Tuple[list[float], list[float]]]) -> None:
    _, ax = plt.subplots(figsize=(12,10))

    handles = []
    labels = []

    # datasets_list = [dataset for dataset in datasets.items()]
    # label_y1, (data_x1, data_y1) = datasets_list.pop()
    # handles.append(plt.scatter(data_x1, data_y1))
    # labels.append(label_y1)

    for label_y, data in datasets.items():
        data_x, data_y = data
        handles.append(plt.scatter(data_x, data_y))
        labels.append(label_y)

    ax.set_title(title)
    ax.set_xlabel(label_x)
    if len(datasets) == 1:
        ax.set_ylabel([k for k in datasets.keys()][0])
    else:
        ax.legend(handles, labels)


if __name__ == '__main__':
    surface_reflectance_csv_file_path = 'filtered_scaled_MOD09A1.csv'
    phenocam_csv_file_path = 'S:\school\geog5225\data_processing\provisional_data\data_record_4\willowcreek_DB_1000_1day.csv'
    pixels_of_interest = [127, 128, 129, 144, 145, 146, 161, 162, 163]

    surface_reflectance_samples = read_samples_from_file(
        surface_reflectance_csv_file_path, 7, pixels_of_interest, 4, 0, parse_acquisition_day=lambda row: int(row[2][5:]), parse_acquisition_year=lambda row: int(row[2][1:5]), parse_product=lambda row: row[1])

    # surface_reflectance_samples.sort(key=lambda sample: sample.acquisition_day)

    phenocam_samples = read_samples_from_file(
        phenocam_csv_file_path, 1, [11, 16], 5, 23, parse_acquisition_day=lambda row: int(row[2]), parse_acquisition_year=lambda row: int(row[1]), parse_product=lambda _: 'PhenoCam')

    # phenocam_samples.sort(key=lambda sample: sample.acquisition_day)

    data_to_graph = {}

    """ Phenocam """

    pheno_x = []
    pheno_gcc = []

    for sample in phenocam_samples:
        pheno_x.append(sample.acquisition_day)
        pheno_gcc.append(sample.pixels[0][0])

    data_to_graph['GCC'] = (pheno_x, pheno_gcc)

    """ Surface Reflectance """

    sr_x = []

    for sample in surface_reflectance_samples:
        sr_x.append(sample.acquisition_day)

    synced_pheno_gcc = []
    for x in sr_x:
        synced_pheno_gcc.append(pheno_gcc[pheno_x.index(x)])

    red_based = False

    veg_indicies = {
        'Green Ratio': (green_ratio, {}, red_based),
        'EVI3': (evi3, {}, red_based),
        'EVI2': (evi2, {}, red_based),
        'WDRVI (a=0.2)': (wdrvi, {'alpha': 0.2}, red_based),
        'Red Ratio': (red_ratio, {}, not red_based),
        'WDRVI (a=0.25)': (wdrvi, {'alpha': 0.25}, red_based),
        'WDRVI (a=0.3)': (wdrvi, {'alpha': 0.3}, red_based),
        'NDVI': (ndvi, {}, red_based),
        'T_CAP': (tasseled_cap_greenness, {}, red_based)
    }

    for i, veg_index in enumerate(veg_indicies.items()):
        index_name, sample_avg_params = veg_index

        print(index_name)

        vi_values = []
        for sample in surface_reflectance_samples:
            vi_values.append(sample_avg(sample, *sample_avg_params))

        print('r^2 =', r_squared(synced_pheno_gcc, vi_values))

        estimation_graph('DOY', sr_x, index_name, vi_values, 'GCC', synced_pheno_gcc)
        correlation_graph(index_name, vi_values, 'GCC', synced_pheno_gcc)
        data_to_graph[index_name] = (sr_x, vi_values)

    standard_graph(f'{VALID_YEARS[0]}-{VALID_YEARS[-1]} Index Values', 'DOY', data_to_graph)

    standard_graph(f'{VALID_YEARS[0]}-{VALID_YEARS[-1]} PhenoCam GCC Values', 'DOY', {'GCC': (pheno_x, pheno_gcc)})

    plt.show()
