#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Programme analyst Dengue Incidence 1990 to 2019.

it has three functions which produce there different charts.


Author: Mohamed Jahbar
"""
# doing all the imports HERE
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns


def processWorldbank_POP_TOT(filename_path):
    """processWorldbank_POP_TOT function.

    This function process the population file.
    filename:contains the file name.
    """
    df_worldbank = pd.read_csv(filename_path, skiprows=3)
    # data cleanning
    # Country Code is needed for future processing
    df_Bycountry_pop = df_worldbank.loc[:, ~df_worldbank.columns.isin([
        'Indicator Code', 'Indicator Name'])]

    # removing 'Indicator Code', 'Indicator Name','Country Code'
    df_ByYear_pop = df_worldbank.loc[:, ~df_worldbank.columns.isin([
        'Indicator Code', 'Indicator Name', 'Country Code'])]
    df_ByYear_pop = df_ByYear_pop.transpose()
    df_ByYear_pop.reset_index(drop=False, inplace=True)
    df_ByYear_pop.columns = df_ByYear_pop.iloc[0]
    # Drop the first row as it's now used for column names
    df_ByYear_pop = df_ByYear_pop[1:]
    # rename column
    df_ByYear_pop = df_ByYear_pop.rename(columns={'Country Name': 'Year'})
    # cleaning: removing last row
    df_ByYear_pop.drop(df_ByYear_pop.index[-1], inplace=True)
    df_ByYear_pop = df_ByYear_pop.set_index('Year')
    # Changing the index to integers
    df_ByYear_pop.index = df_ByYear_pop.index.astype(int)

    return df_Bycountry_pop, df_ByYear_pop


def processDengueIncidence(filename):
    """processDengueIncidence function.

    This function process the dengueIncidence file.
    filename:contains the file name.
    """
    # Reading csv file into dataframe
    df_dengueIncidence = pd.read_csv(filename)
    # Making Year colume as index
    df_dengueIncidence = df_dengueIncidence.set_index('Years')
    # Return by rounding the numbers
    return df_dengueIncidence.round()


def getTop5countriesDengueIncidence(df_DengueIncidence):
    """getTop5countriesDengueIncidence function.

    This function retruns Top 5 countries with Dengue Incidence
    df_DengueIncidence: A dataframe contains dengue Incidences by countries.
    """
    growth_ratesList = []

    # Calculate growth rates for each country
    for column_name, column_data in df_DengueIncidence.iteritems():
        data = []
        data.append(column_name)
        if (column_data.iloc[-1] - column_data.iloc[0]) == 0:
            continue

        absolute_change = column_data.iloc[-1] - column_data.iloc[0]
        growth_rates = (column_data.iloc[-1] -
                        column_data.iloc[0]) / column_data.iloc[0] * 100

        data.append((round(growth_rates, 2)))
        data.append((round(absolute_change, 2)))

        growth_ratesList.append(data)

    # Create the pandas DataFrame
    df_growth_rates = pd.DataFrame(growth_ratesList, columns=[
        'Country', 'dengueIncidence_growth_rates%', 'absolute_change'])
    df_growth_rates = df_growth_rates.sort_values([
        'absolute_change', 'dengueIncidence_growth_rates%'], ascending=False)

    # Select the top 5 countries with the highest values
    top_5_countries = df_growth_rates.head(5)

    # Display the top 5 countries
    return top_5_countries


def plotTop5countriesPopulation(pop_byYear, top_5_countries):
    """plotTop5countriesPopulation function.

    This function plot Top 5 countries Population
    pop_byYear: A dataframe contains population by year.
    top_5_countries : A dataframe contains Top 5 dengue Incidences by countries
    """
    years = pop_byYear.index
    country_names = pop_byYear.columns
    # change to population to billion for ease of calulation

    population_data = pop_byYear

    # Filter the DataFrame based on the index range between 1990 to 2019
    filtered_df = population_data[(population_data.index >= 1990)
                                  & (population_data.index <= 2019)]
    filtered_df = filtered_df[top_5_countries['Country']]
    # Plotting the top 5 countries with highest growth rates
    plt.figure(figsize=(10, 6))

    for i in range(5):
        plt.plot(filtered_df.index, filtered_df.iloc[:, [i]], marker='o',
                 linestyle='-', label=filtered_df.columns[i])

    # Adding labels and title
    plt.xlabel('Year')
    plt.ylabel('Human Population (in billions)')
    plt.title('Population Growth Rates')

    # Show legend
    plt.legend()

    # Show grid
    plt.grid(True)

    # Show the plot
    plt.show()


def plotComparisonDengueIncidenceGrowthRates(top_5_countries):
    """plotComparisonDengueIncidenceGrowthRates function.

    This function plot comparison of Dengue Incidence Growth Rates
    top_5_countries : A dataframe contains Top 5 dengue Incidences by countries
    """
    # Create a figure and axis
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot 'growth_rates%' as a line plot
    ax1.plot(top_5_countries['Country'], top_5_countries[
        'dengueIncidence_growth_rates%'], marker='o',
        linestyle='-', color='orange', label='Growth Rates (%)')
    ax1.set_xlabel('Country')
    ax1.set_ylabel('Growth Rates (%)', color='orange')
    ax1.tick_params(axis='x', rotation=45)
    ax1.legend(loc='upper left')

    # Create a secondary y-axis for 'absolute_change'
    ax2 = ax1.twinx()
    ax2.bar(top_5_countries['Country'], top_5_countries[
        'absolute_change'], color='skyblue', alpha=0.5,
        label='Absolute Change')
    ax2.set_ylabel('Absolute Change', color='skyblue')
    ax2.legend(loc='upper right')

    # Set title
    plt.title('Comparison of Dengue Incidence Growth ' +
              'Rates and Absolute Change in Countries')

    # Show the plot
    plt.tight_layout()
    plt.show()


def processSurfaceTemp(surface_temperature_file_PH, master_Philippines):
    """processSurfaceTemp function.
    This function process Surface Temperature data
    master_Philippines : A dataframe contains population,
    and Dengue Incidence Growth Rates
    """
    # reading from surface temperature file of PH
    df_surface_temperature_PH = pd.read_csv(surface_temperature_file_PH)
    # droping columns
    df_surface_temperature_PH.drop(df_surface_temperature_PH.iloc[:, 0:9],
                                   axis=1, inplace=True)
    # Transpose data by year
    df_surface_temperature_PH = df_surface_temperature_PH.transpose()
    # Change colume to int data type
    df_surface_temperature_PH.index = df_surface_temperature_PH.index.astype(
        int)
    # filter data between year 1990 to 2019
    df_surface_temperature_PH = df_surface_temperature_PH[(
        df_surface_temperature_PH.index >= 1990) & (
            df_surface_temperature_PH.index <= 2019)]

    # rename column to surface_temp
    df_surface_temperature_PH.columns = ['surface_temp']
    master_Philippines = pd.concat([
        master_Philippines, df_surface_temperature_PH], axis=1)

    return master_Philippines


def plotPopGrowthVsChangeSurfaceTemp(master_Philippines):
    """plotPopGrowthVsChangeSurfaceTemp function.

    This plot Population Growth vs Change in Surface Temperature'
    master_Philippines : A dataframe contains population,
    ,Dengue Incidence Growth Rates and Surface Temperture
    """
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Population Growth on primary y-axis (left)
    ax1.plot(master_Philippines.index, master_Philippines[
        'population'], color='blue', marker='o', label='Population Growth')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Population Growth Rate (billion)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.legend(loc='upper left')

    # Surface Temperature on secondary y-axis (right)
    ax2 = ax1.twinx()
    ax2.plot(master_Philippines.index, master_Philippines[
        'surface_temp'], color='red', marker='x', label='Surface Temperature')
    ax2.set_ylabel('Change in Surface Temperature ($^\circ$C)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.legend(loc='upper right')

    plt.title('Population Growth vs Change in Surface Temperature')
    plt.grid(True)
    plt.show()


def processForst_area(forst_area_file, master_Philippines,
                      df_DengueIncidence):
    """processForst_area function.

     This function process Forst_area data
     master_Philippines : A dataframe contains population,
     Dengue Incidence Growth Rates, suface temperature
     """
    # reading from forst_area
    df_forst_area = pd.read_csv(forst_area_file)

    # droping columns
    df_forst_area.drop(df_forst_area.iloc[:, 1:4], axis=1, inplace=True)
    # transpose the dataset
    df_forst_area = df_forst_area.transpose()
    df_forst_area.reset_index(drop=False, inplace=True)
    df_forst_area.columns = df_forst_area.iloc[0]

    # Drop the first row as it's now used for column names
    df_forst_area = df_forst_area[1:]

    # rename column
    df_forst_area = df_forst_area.rename(columns={'Country Name': 'Year'})
    # cleaning: removing last row
    df_forst_area.drop(df_forst_area.index[-1], inplace=True)
    # set index
    df_forst_area = df_forst_area.set_index('Year')

    # Changing the index to integers
    df_forst_area.index = df_forst_area.index.astype(int)

    df_forst_area.index = df_forst_area.index.astype(int)

    df_forst_area_years = df_forst_area[(
        df_forst_area.index >= 1990) & (
            df_forst_area.index <= 2019)]['Philippines']

    master_Philippines = pd.concat([
        master_Philippines, df_forst_area_years], axis=1)
    # rename column
    master_Philippines = master_Philippines.rename(
        columns={'Philippines': 'de_forst_area'})

    # convert column "ade_forst_area" to float
    # using apply method
    master_Philippines[['de_forst_area']] = master_Philippines[
        ['de_forst_area']].apply(pd.to_numeric)

    master_Philippines = pd.concat([
        master_Philippines, df_DengueIncidence['Philippines']], axis=1)
    master_Philippines = master_Philippines.rename(
        columns={'Philippines': 'dengueIncidence'})

    return master_Philippines


def plotCorelationCoeffiecnt(master_Philippines):
    """plotCorelationCoeffiecnt function.

    This plot the Corelation Coeffiecnt de_forst_area and dengueIncidence
    master_Philippines : A dataframe contains population,
    ,Dengue Incidence Growth Rates and Surface Temperture
    """
    # Extracting columns for analysis
    de_forst_area = master_Philippines['de_forst_area']
    dengue_incidence = master_Philippines['dengueIncidence']

    # Calculating correlation coefficient
    correlation_matrix = np.corrcoef(de_forst_area, dengue_incidence)
    correlation_coefficient = correlation_matrix[0, 1]
    print(correlation_matrix)

    # Plotting the correlation coefficient
    plt.figure(figsize=(6, 6))
    plt.scatter(de_forst_area, dengue_incidence, alpha=0.7)
    plt.title(f'Correlation Coefficient: {correlation_coefficient:.2f}')
    plt.xlabel('Deforestation Area')
    plt.ylabel('Dengue Incidence')
    plt.grid(True)
    plt.show()


def processHealthcareAccessAndQualityIndex(healthcareAccessAndQualityIndex_file, 
                                           master_Philippines):
    """processHealthcareAccessAndQualityIndex function.

     This function process Healthcare Access And Quality Index
     master_Philippines : A dataframe contains population,
     Dengue Incidence Growth Rates, suface temperature
     """
    # reading from Healthcare Access And Quality Index
    df_healthcareAccessAndQualityIndex = pd.read_csv(
        healthcareAccessAndQualityIndex_file)
    # print data frame
    print(df_healthcareAccessAndQualityIndex.info())

    # Fillter and extract data for Philippines
    df_healthcareAccessAndQualityIndex = df_healthcareAccessAndQualityIndex.loc[
        df_healthcareAccessAndQualityIndex['Entity'] == "Philippines", [
            "Entity", "Year", "HAQ Index (IHME (2017))"]]
    df_healthcareAccessAndQualityIndex = df_healthcareAccessAndQualityIndex.set_index('Year')
    master_Philippines = pd.concat([master_Philippines,
                                    df_healthcareAccessAndQualityIndex[
                                        "HAQ Index (IHME (2017))"]], axis=1)

    return master_Philippines



def print_stats(master_Philippines):
    """print_stats function.

     This function print stats of dataframe
     master_Philippines : A dataframe contains population,
     Dengue Incidence Growth Rates, suface temperature and
     healthcareAccessAndQualityIndex
     """
    print_stats_lambda = lambda data: {
        'max': np.max(data),
        'std_dev': np.std(data),
        'skewness': stats.skew(data),
        'kurtosis': stats.kurtosis(data)
    }

    # Loop through columns using a for loop
    for col in master_Philippines.columns:
        print(f'{col}:{print_stats_lambda(master_Philippines[col].values.tolist())}')


def plotCompare_HAQ_index_dengue_incidence(master_Philippines):
    """plotCorelationCoeffiecnt function.

    This plot the compare HAQ index and dengue_incidence
    master_Philippines : A dataframe contains population,
    ,Dengue Incidence Growth Rates and Surface Temperture
    """
    compare_HAQ_index_dengue_incidence = master_Philippines.loc[
        [1990, 1995, 2000, 2005, 2010, 2015]]
    years = compare_HAQ_index_dengue_incidence.index
    dengue_incidence = compare_HAQ_index_dengue_incidence["dengueIncidence"]
    haq_index = compare_HAQ_index_dengue_incidence["HAQ Index (IHME (2017))"]

    # Plotting
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Bar chart for HAQ Index
    ax1.bar(years, haq_index, color='blue', alpha=0.5,
            label='HAQ Index (IHME 2017)')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('HAQ Index', color='blue')
    ax1.tick_params('y', colors='blue')

    # Line chart for Dengue Incidence
    ax2 = ax1.twinx()
    ax2.plot(years, dengue_incidence, marker='o', linestyle='-',
             color='red', label='Dengue Incidence')
    ax2.set_ylabel('Dengue Incidence', color='red')
    ax2.tick_params('y', colors='red')

    # Title and legend
    plt.title('Dengue Incidence vs HAQ Index (IHME 2017) Over Time')
    plt.legend(loc='upper left')

    # Show plot
    plt.tight_layout()
    plt.show()


def plotHeatMapCorrelationmatrix(master_Philippines):
    """plotHeatMap function.

    This plot HeatMap of Correlation matrix
    master_Philippines : A dataframe contains population,
    ,Dengue Incidence Growth Rates and Surface Temperture
    """
    # Compute the correlation matrix
    corr_matrix = master_Philippines.corr()

    # Create a heatmap of the correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm',
                square=True, fmt=".2f", annot_kws={"size": 10})
    plt.title('Correlation Matrix Plot')
    plt.show()


def main():
    """Main function is the entry point for the application.
    data taken from:
    Socioeconomic Data: Population:API_SP.POP.TOTL_DS2_en_csv_v2_6011311.csv
                        https://data.worldbank.org/indicator/SP.POP.TOTL?view=chart
    Dengue Incidence Data: dengue-incidence.csv
                        https://ourworldindata.org/grapher/dengue-incidence?tab=table
    healthcare-access-and-quality-index file:
        healthcare-access-and-quality-index.csv
        https://ourworldindata.org/grapher/healthcare-access-and-quality-index?tab=table&time=1990..latest&country=
    surface temperature:Annual_Surface_Temperature_Change.csv
    https://climatedata.imf.org/pages/climatechange-data
    Forst Area :API_AG.LND.FRST.ZS_DS2_en_csv_v2_5994693.csv
    https://data.worldbank.org/indicator/AG.LND.FRST.ZS?end=2017&start=1990&view=chart
    preforming data pre-processing before passing into functions
    """

    # Assinging the filename to variables
    worldbank_POP_TOTL_file = "API_SP.POP.TOTL_DS2_en_csv_v2_6011311.csv"
    dengueIncidence_file = "dengue_incidence_processed.csv"
    surface_temperature_file_PH = "Annual_Surface_Temperature_Change_PH.csv"
    forst_area_file = "API_AG.LND.FRST.ZS_DS2_en_csv_v2_5994693.csv"
    healthcareAccQIndex_file = "healthcare-access-and-quality-index.csv"

    # get population data
    pop_byCountry, pop_byYear = processWorldbank_POP_TOT(
                                worldbank_POP_TOTL_file)

    # get dengue incidence of top 5 countries and print it
    df_DengueIncidence = processDengueIncidence(dengueIncidence_file)
    top_5_countries = getTop5countriesDengueIncidence(df_DengueIncidence)
    print(top_5_countries)

    # plot Top 5 countries Population
    plotTop5countriesPopulation(pop_byYear, top_5_countries)

    # plot Comparison Dengue Incidence GrowthRates
    plotComparisonDengueIncidenceGrowthRates(top_5_countries)

    # combine the data by top 5 countries
    # Filter the DataFrame based on the index range
    filtered_df = pop_byYear[(pop_byYear.index >= 1990) &
                             (pop_byYear.index <= 2019)]
    filtered_df = filtered_df[top_5_countries['Country']]

    # Take Philippines for further analisis

    master_Philippines = filtered_df[['Philippines']].copy()
    master_Philippines.columns = ['population']

    master_Philippines = processSurfaceTemp(
        surface_temperature_file_PH, master_Philippines)
    # plot Population Growth vs Change in Surface Temperature
    plotPopGrowthVsChangeSurfaceTemp(master_Philippines)
    #
    master_Philippines = processForst_area(
        forst_area_file, master_Philippines, df_DengueIncidence)
    # plot the Corelation Coeffiecnt de_forst_area and dengueIncidence
    plotCorelationCoeffiecnt(master_Philippines)
    # process Healthcare Access And QualityIndex
    master_Philippines = processHealthcareAccessAndQualityIndex(
        healthcareAccQIndex_file, master_Philippines)
    # print statistic data of dataframe
    print(master_Philippines.describe())
    print_stats(master_Philippines)
    # plot to Compare HAQ_index to dengue incidence
    plotCompare_HAQ_index_dengue_incidence(master_Philippines)
    # plot Correlation matrix
    plotHeatMapCorrelationmatrix(master_Philippines)



if __name__ == "__main__":
    main()