async function fetchAndRenderPlot(url, containerId) {
    try {
        const response = await fetch(url);
        const plotJson = await response.json();
        const plotData = JSON.parse(plotJson); // Parse the inner JSON string
        Plotly.newPlot(containerId, plotData.data, plotData.layout);
    } catch (error) {
        console.error(`Error fetching and rendering plot for ${containerId}:`, error);
        document.getElementById(containerId).innerHTML = `<p class="text-danger">Failed to load chart.</p>`;
    }
}

async function fetchAndRenderDemographicPlots(url, genderContainerId, seniorCitizenContainerId) {
    try {
        const response = await fetch(url);
        const plotsJson = await response.json();
        
        const genderPlotData = JSON.parse(plotsJson.gender);
        Plotly.newPlot(genderContainerId, genderPlotData.data, genderPlotData.layout);

        const seniorCitizenPlotData = JSON.parse(plotsJson.seniorCitizen);
        Plotly.newPlot(seniorCitizenContainerId, seniorCitizenPlotData.data, seniorCitizenPlotData.layout);

    } catch (error) {
        console.error(`Error fetching and rendering demographic plots:`, error);
        document.getElementById(genderContainerId).innerHTML = `<p class="text-danger">Failed to load gender chart.</p>`;
        document.getElementById(seniorCitizenContainerId).innerHTML = `<p class="text-danger">Failed to load senior citizen chart.</p>`;

    }
}

function createPlots() {
    fetchAndRenderPlot('/api/eda/churn_distribution', 'churnDistribution');
    fetchAndRenderPlot('/api/eda/tenure_vs_churn', 'tenureAnalysis');
    fetchAndRenderPlot('/api/eda/monthly_charges_impact', 'monthlyCharges');
    fetchAndRenderPlot('/api/eda/service_impact', 'serviceImpact');
    fetchAndRenderPlot('/api/eda/contract_type_analysis', 'contractAnalysis');
    fetchAndRenderPlot('/api/eda/payment_method_analysis', 'paymentAnalysis');
    fetchAndRenderPlot('/api/eda/internet_service_analysis', 'internetAnalysis');
    fetchAndRenderDemographicPlots('/api/eda/demographic_patterns', 'demographicAnalysis', 'demographicAnalysis'); // Reusing same div for simplicity, will add more divs if needed.
    fetchAndRenderPlot('/api/eda/additional_services_impact', 'additionalServices');
    fetchAndRenderPlot('/api/eda/overall_churn_factors', 'churnFactors');
} 