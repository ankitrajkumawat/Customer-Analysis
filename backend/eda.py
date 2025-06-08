import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os

class EDAProcessor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = self._load_and_clean_data()
        self.output_dir = os.path.join('frontend', 'static', 'images', 'eda')
        os.makedirs(self.output_dir, exist_ok=True)

    def _load_and_clean_data(self):
        df = pd.read_csv(self.data_path)
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
        df = df.drop('customerID', axis=1)
        return df

    def _save_plot(self, fig, filename):
        filepath = os.path.join(self.output_dir, filename)
        fig.write_image(filepath)
        print(f"Saved {filename} to {filepath}")

    def get_churn_distribution_plot(self):
        churn_counts = self.df['Churn'].value_counts(normalize=True) * 100
        fig = px.bar(
            x=churn_counts.index,
            y=churn_counts.values,
            title='Distribution of Customer Churn',
            labels={'x': 'Churn Status', 'y': 'Percentage of Customers'},
            color=churn_counts.index,
            color_discrete_map={'Yes': '#e74c3c', 'No': '#28a745'}
        )
        fig.update_layout(title_x=0.5, yaxis_title='Percentage', xaxis_title='Churn Status', margin=dict(b=120))
        return fig

    def get_tenure_vs_churn_plot(self):
        df_churn_tenure = self.df.groupby(['tenure', 'Churn']).size().unstack(fill_value=0)
        df_churn_tenure['Churn Rate'] = (df_churn_tenure['Yes'] / (df_churn_tenure['Yes'] + df_churn_tenure['No'])) * 100
        
        fig = px.line(
            df_churn_tenure.reset_index(),
            x='tenure',
            y='Churn Rate',
            title='Churn Rate by Customer Tenure',
            labels={'tenure': 'Tenure (Months)', 'Churn Rate': 'Churn Rate (%)'}
        )
        fig.update_layout(title_x=0.5, yaxis_title='Churn Rate (%)', xaxis_title='Tenure (Months)', margin=dict(b=120))
        return fig

    def get_monthly_charges_impact_plot(self):
        fig = px.box(
            self.df,
            x='Churn',
            y='MonthlyCharges',
            color='Churn',
            title='Monthly Charges vs. Churn',
            labels={'MonthlyCharges': 'Monthly Charges', 'Churn': 'Churn Status'},
            color_discrete_map={'Yes': '#e74c3c', 'No': '#28a745'}
        )
        fig.update_layout(title_x=0.5, margin=dict(b=120))
        return fig

    def get_service_impact_plot(self):
        services_cols = [
            'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
            'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'
        ]
        
        churn_rates = []
        for col in services_cols:
            churn_rate = self.df.groupby(col)['Churn'].value_counts(normalize=True).unstack().get('Yes', pd.Series(0))
            for idx, val in churn_rate.items():
                churn_rates.append({'Service': col, 'Option': idx, 'Churn Rate': val * 100})
        
        churn_rates_df = pd.DataFrame(churn_rates)
        
        fig = px.bar(
            churn_rates_df,
            x='Service',
            y='Churn Rate',
            color='Option',
            barmode='group',
            title='Churn Rate by Service Type',
            labels={'Churn Rate': 'Churn Rate (%)'}
        )
        fig.update_layout(title_x=0.5, yaxis_title='Churn Rate (%)', xaxis_title='Service Type', margin=dict(b=120))
        return fig

    def get_contract_type_analysis_plot(self):
        contract_churn = self.df.groupby('Contract')['Churn'].value_counts(normalize=True).unstack()
        contract_churn['Churn Rate'] = contract_churn['Yes'] * 100
        
        fig = px.bar(
            contract_churn.reset_index(),
            x='Contract',
            y='Churn Rate',
            color='Contract',
            title='Churn Rate by Contract Type',
            labels={'Churn Rate': 'Churn Rate (%)'},
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig.update_layout(title_x=0.5, yaxis_title='Churn Rate (%)', xaxis_title='Contract Type', margin=dict(b=120))
        return fig

    def get_payment_method_analysis_plot(self):
        payment_churn = self.df.groupby('PaymentMethod')['Churn'].value_counts(normalize=True).unstack()
        payment_churn['Churn Rate'] = payment_churn['Yes'] * 100
        
        fig = px.bar(
            payment_churn.reset_index(),
            x='PaymentMethod',
            y='Churn Rate',
            color='PaymentMethod',
            title='Churn Rate by Payment Method',
            labels={'Churn Rate': 'Churn Rate (%)'},
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig.update_layout(title_x=0.5, yaxis_title='Churn Rate (%)', xaxis_title='Payment Method', margin=dict(b=120))
        return fig

    def get_internet_service_analysis_plot(self):
        internet_churn = self.df.groupby('InternetService')['Churn'].value_counts(normalize=True).unstack()
        internet_churn['Churn Rate'] = internet_churn['Yes'] * 100
        
        fig = px.bar(
            internet_churn.reset_index(),
            x='InternetService',
            y='Churn Rate',
            color='InternetService',
            title='Churn Rate by Internet Service Type',
            labels={'Churn Rate': 'Churn Rate (%)'},
            color_discrete_sequence=px.colors.qualitative.Dark2
        )
        fig.update_layout(title_x=0.5, yaxis_title='Churn Rate (%)', xaxis_title='Internet Service Type', margin=dict(b=120))
        return fig

    def get_demographic_gender_plot(self):
        fig = px.bar(
            self.df.groupby('gender')['Churn'].value_counts(normalize=True).unstack().get('Yes', pd.Series(0)).reset_index(),
            x='gender',
            y='Yes',
            title='Churn Rate by Gender',
            labels={'Yes': 'Churn Rate (%)', 'gender': 'Gender'},
            color_discrete_map={'Male': '#3498db', 'Female': '#e74c3c'}
        )
        fig.update_layout(yaxis_title='Churn Rate (%)', title_x=0.5, margin=dict(b=120))
        return fig

    def get_demographic_senior_citizen_plot(self):
        fig = px.bar(
            self.df.groupby('SeniorCitizen')['Churn'].value_counts(normalize=True).unstack().get('Yes', pd.Series(0)).reset_index(),
            x='SeniorCitizen',
            y='Yes',
            title='Churn Rate by Senior Citizen Status',
            labels={'Yes': 'Churn Rate (%)', 'SeniorCitizen': 'Senior Citizen (0=No, 1=Yes)'},
            color_discrete_map={0: '#28a745', 1: '#ffc107'}
        )
        fig.update_layout(yaxis_title='Churn Rate (%)', title_x=0.5, margin=dict(b=120))
        return fig

    def get_additional_services_impact_plot(self):
        additional_services = [
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
            'TechSupport', 'StreamingTV', 'StreamingMovies'
        ]

        churn_rates = []
        for service in additional_services:
            churn_rate = self.df.groupby(service)['Churn'].value_counts(normalize=True).unstack().get('Yes', pd.Series(0))
            for status, rate in churn_rate.items():
                if status in ['Yes', 'No']:
                    churn_rates.append({'Service': service, 'Status': status, 'Churn Rate': rate * 100})
        
        df_plot = pd.DataFrame(churn_rates)
        
        fig = px.bar(
            df_plot,
            x='Service',
            y='Churn Rate',
            color='Status',
            barmode='group',
            title='Churn Rate by Additional Service Status',
            labels={'Churn Rate': 'Churn Rate (%)'},
            color_discrete_map={'Yes': '#e74c3c', 'No': '#28a745'}
        )
        fig.update_layout(title_x=0.5, yaxis_title='Churn Rate (%)', xaxis_title='Additional Service', margin=dict(b=120))
        return fig

    def get_overall_churn_factors_plot(self):
        # For simplicity, let's calculate churn rate for a few key categorical features
        # You can expand this to include more features or use feature importance from models
        features = [
            'Contract', 'InternetService', 'PaymentMethod', 
            'gender', 'Partner', 'Dependents', 'SeniorCitizen'
        ]
        
        churn_data = []
        for feature in features:
            churn_counts = self.df.groupby(feature)['Churn'].value_counts(normalize=True).unstack()
            if 'Yes' in churn_counts.columns:
                for index, row in churn_counts.iterrows():
                    churn_rate = row.get('Yes', 0) * 100
                    churn_data.append({'Feature': feature, 'Category': str(index), 'Churn Rate': churn_rate})

        df_plot = pd.DataFrame(churn_data)

        fig = px.bar(
            df_plot,
            x='Feature',
            y='Churn Rate',
            color='Category',
            barmode='group',
            title='Overall Churn Rate by Key Factors',
            labels={'Churn Rate': 'Churn Rate (%)', 'Feature': 'Feature'},
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig.update_layout(title_x=0.5, yaxis_title='Churn Rate (%)', xaxis_title='Key Factor', margin=dict(b=120))
        return fig

    def generate_and_save_all_plots(self):
        plots = {
            'churn_distribution.png': self.get_churn_distribution_plot(),
            'tenure_vs_churn.png': self.get_tenure_vs_churn_plot(),
            'monthly_charges_impact.png': self.get_monthly_charges_impact_plot(),
            'service_impact.png': self.get_service_impact_plot(),
            'contract_type_analysis.png': self.get_contract_type_analysis_plot(),
            'payment_method_analysis.png': self.get_payment_method_analysis_plot(),
            'internet_service_analysis.png': self.get_internet_service_analysis_plot(),
            'gender_analysis.png': self.get_demographic_gender_plot(),
            'senior_citizen_analysis.png': self.get_demographic_senior_citizen_plot(),
            'additional_services_impact.png': self.get_additional_services_impact_plot(),
            'overall_churn_factors.png': self.get_overall_churn_factors_plot()
        }

        for filename, fig in plots.items():
            self._save_plot(fig, filename)


if __name__ == '__main__':
    # Get the path to the dataset
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(os.path.dirname(current_dir), 'WA_Fn-UseC_-Telco-Customer-Churn.csv')
    
    eda_processor = EDAProcessor(data_path)
    eda_processor.generate_and_save_all_plots()
    print("All EDA charts generated and saved as images.") 