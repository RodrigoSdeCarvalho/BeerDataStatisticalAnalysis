import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

#Excel viewer extension in vscode is recommended to view the csv files.

results_path = os.path.join(os.getcwd(), "results")

def analyze_beer_data(brand:str, filtered_column:str, x_label:str,  show:bool = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Plots a histogram and a boxplot of the filtered_column of the brand beer data.
    Also, calculates the statistics of the filtered_column of the brand beer data and 
    detecs the outliers of the filtered_column of the brand beer data.

    Args:
        brand (str): brand of the beer data to be analyzed.
        filtered_column (str): product column of the beer data to be analyzed.
        x_label (str): x label (and title) of the histogram and boxplot.
        show (bool, optional): Determines if the histogram and boxplot will be shown. Defaults to False.

    Returns:
        tuple: dataframe of the beer prices, dataframes of the statistics and outliers of the filtered_column of the brand beer data.
    """
    beer_df = get_beer_df(brand=brand, filtered_column=filtered_column, excel_file_name="table.xlsx")
    plot_beer_df_hist(beer_df=beer_df, x_label=x_label, show=show)
    beer_df_stats = calculate_beer_df_stats(beer_df=beer_df, file_name=f"{brand}_stats")
    beer_df_df_outliers = get_beer_df_outliers(beer_df=beer_df, filtered_column=filtered_column, file_name=f"{brand}_outliers")

    return beer_df, beer_df_stats, beer_df_df_outliers


def get_beer_df(brand:str, filtered_column:str, excel_file_name:str) -> pd.DataFrame:
    """Gets the beer data of the brand and filters the data by the filtered_column (product).

    Args:
        brand (str): brand of the beer data to be analyzed
        filtered_column (str): product column of the beer data to be analyzed.
        excel_file_name (str): name of the file that contains the beer data.

    Returns:
        pd.DataFrame: dataframe of the prices of the selected product of the brand.
    """
    beer_table = pd.read_excel(excel_file_name)
    beer_df = beer_table[(beer_table['Marca'] == brand) & (beer_table[filtered_column] > 0)]
    beer_df = beer_df[[filtered_column]]

    return beer_df


def plot_beer_df_hist(beer_df:pd.DataFrame, x_label:str, show:bool = False) -> None:
    """Plots a histogram of the beer_df. And saves the histogram in a file.

    Args:
        beer_df (pd.DataFrame): dataframe of the prices of the selected product of the brand.
        x_label (str): x label (and title) of the histogram.
        show (bool, optional): Determines if the histogram will be shown. Defaults to False.
    """
    plt.hist(beer_df, color='green', edgecolor='black', cumulative=False, range=(0,8), bins=20, density=True)
    plt.xlabel(x_label)
    plt.ylabel("Quantidade")
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.grid(True, axis='y')
    fig_path = os.path.join(results_path, f"{x_label}.png")
    plt.savefig(fig_path)

    if show:
        plt.show() 
        
    plt.close()


def calculate_beer_df_stats(beer_df:pd.DataFrame, file_name:str) -> pd.DataFrame:
    """Calculates the statistics of the beer_df:
    mean, mode, median, variance, standard deviation, standard error,
    variant coefficient, skewness and kurtosis. And saves the statistics in a csv file.

    Args:
        beer_df (pd.DataFrame): dataframe of the prices of the selected product of the brand.
        file_name (str): name of the file that will contain the statistics.

    Returns:
        pd.DataFrame: dataframe of the statistics of the beer_df.
    """
    mean = beer_df.mean().iloc[0]
    mode = beer_df.mode().iloc[0][0]
    median = beer_df.median()[0]
    variance = beer_df.var()[0]
    std = beer_df.std()[0]
    standard_mean_error = beer_df.sem()[0]
    variant_coefficient = std / mean
    assymetry = beer_df.skew()[0]
    kurtose = beer_df.kurtosis()[0]  

    beer_df_stats = {
        "Cerveja": [file_name],
        "mean": [mean],
        "mode": [mode],
        "median": [median],
        "variance": [variance],
        "std": [std],
        "standard_mean_error": [standard_mean_error],
        "variant_coefficient": [variant_coefficient],
        "assymetry": [assymetry],
        "kurtose": [kurtose]
    }

    beer_df_stats = pd.DataFrame(beer_df_stats)
    beer_stats_csv_path = os.path.join(results_path, f"{file_name}.csv")
    beer_df_stats.to_csv(beer_stats_csv_path, index=False)

    return beer_df_stats


def plot_beer_df_boxplot(beer_dfs:list[pd.DataFrame], x_labels:list[str], show:bool = False) -> None:
    """Plots a boxplot of the beer_df. And saves the boxplot in a file.

    Args:
        beer_df (pd.DataFrame): dataframe of the prices of the selected product of the brand.
        x_label (str): x label (and title) of the boxplot.
        show (bool, optional): Determines if the boxplot will be shown. Defaults to False.
    """
    for i in range(len(beer_dfs)):
        beer_dfs[i] = beer_dfs[i].to_numpy().flatten()

    x_label = f"{x_labels[0]} x {x_labels[1]}"
    plt.boxplot(beer_dfs, labels=x_labels)
    plt.xlabel(x_label)
    plt.grid(True, axis='y')
    fig_path = os.path.join(results_path, f"boxplot_{x_label}.png")
    plt.savefig(fig_path)

    if show:
        plt.show()

    plt.close()


def get_beer_df_outliers(beer_df:pd.DataFrame, filtered_column:str, file_name:str) -> pd.DataFrame:
    """Detects the outliers of the beer_df. And saves the outliers in a csv file.

    Args:
        beer_df (pd.DataFrame): dataframe of the prices of the selected product of the brand.
        filtered_column (str): product column of the beer data to be analyzed.
        file_name (str): name of the file that will contain the outliers.

    Returns:
        pd.DataFrame: dataframe of the outliers of the beer_df.
    """
    q1 = beer_df.quantile(0.25)[0]
    q3 = beer_df.quantile(0.75)[0]
    iqr = q3 - q1
    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)

    beer_df = beer_df[[filtered_column]]
    beer_df_outliers = beer_df[(beer_df < lower_bound) | (beer_df > upper_bound)]
    beer_df_outliers = beer_df_outliers.dropna()

    beer_outliers_path = os.path.join(results_path, f"{file_name}.csv")
    beer_df_outliers.to_csv(beer_outliers_path, index=False)

    return beer_df_outliers


def compare_beers_data(beer_dfs:list[pd.DataFrame], x_labels:list[str], beers_stats:list, beers_outliers:list, brands:list, show:bool = False) -> None:
    """Compares the statistics and outliers of the beers. And saves the comparisons in csvs files.

    Args:
        beers_stats (list): list of dataframes. List of the statistics of the beers.
        beers_outliers (list): list of dataframes. List of the outliers of the beers.
        brands (list): list of strings. List of the brands of the beers.
    """
    compare_beers_stats(beers_stats=beers_stats, brands=brands)
    compare_beers_outliers(beers_outliers=beers_outliers, brands=brands)
    plot_beer_df_boxplot(beer_dfs=beer_dfs, x_labels=x_labels, show=show)


def compare_beers_stats(beers_stats:list, brands) -> None:
    """Compares the statistics of the beers. And saves the comparisons in csvs files.

    Args:
        beers_stats (list): list of dataframes. List of the statistics of the beers.
        brands (_type_): list of strings. List of the brands of the beers.
    """
    df_beer_stats_comparison = pd.concat([beers_stats[0], beers_stats[1]], ignore_index=True)
    beer_stats_comparison_csv_path = os.path.join(results_path, "beer_stats_comparison.csv")
    df_beer_stats_comparison.to_csv(beer_stats_comparison_csv_path, index=False)


def compare_beers_outliers(beers_outliers:list, brands:list) -> None:
    """Compares the outliers of the beers. And saves the comparisons in csvs files.

    Args:
        beers_outliers (list): list of dataframes. List of the outliers of the beers.
        brands (list): list of strings. List of the brands of the beers.
    """
    beers_outliers[0] = beers_outliers[0].transpose()
    beers_outliers[1] = beers_outliers[1].transpose()
    beers_outliers[0].insert(0, "Cerveja", brands[0])
    beers_outliers[1].insert(0, "Cerveja", brands[1])

    df_beer_outliers_comparison = pd.concat([beers_outliers[0], beers_outliers[1]], ignore_index=True)
    beer_outliers_comparison_path = os.path.join(results_path, "beer_outliers_comparison.csv")
    df_beer_outliers_comparison.to_csv(beer_outliers_comparison_path, index=False)


#First beer
brand_1 = "AMSTEL LAGER"
product_1 = "C16 - LATA – 350 ml"
x_label_1 = "AMSTEL LAGER: LATA – 350 ml"


#Second beer
brand_2 = "BECKS"
product_2 = "C16 - LATA – 350 ml"
x_label_2 = "BECKS: LATA – 350 ml"


if __name__ == "__main__":
    #Comparing the beers
    beer_1_df, beer_1_df_stats, beer_1_df_outliers = analyze_beer_data(brand=brand_1, filtered_column=product_1, x_label=x_label_1, show=False)
    beer_2_df, beer_2_df_stats, beer_2_df_outliers = analyze_beer_data(brand=brand_2, filtered_column=product_2, x_label=x_label_2, show=False)
    compare_beers_data(beer_dfs = [beer_1_df, beer_2_df], x_labels = [x_label_1, x_label_2], beers_stats=[beer_1_df_stats, beer_2_df_stats], 
                       beers_outliers=[beer_1_df_outliers, beer_2_df_outliers], 
                       brands=[brand_1, brand_2])
