{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "8d040402-f208-49ee-b838-950240653539",
   "metadata": {
    "collapsed": true,
    "jupyter": {
     "outputs_hidden": true
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: pandas in c:\\users\\serri\\documents\\coding\\lib\\site-packages (2.0.3)\n",
      "Requirement already satisfied: plotly in c:\\users\\serri\\documents\\coding\\lib\\site-packages (5.9.0)\n",
      "Requirement already satisfied: numpy in c:\\users\\serri\\documents\\coding\\lib\\site-packages (1.24.3)\n",
      "Requirement already satisfied: ipywidgets in c:\\users\\serri\\documents\\coding\\lib\\site-packages (8.0.4)\n",
      "Requirement already satisfied: python-dateutil>=2.8.2 in c:\\users\\serri\\documents\\coding\\lib\\site-packages (from pandas) (2.8.2)\n",
      "Requirement already satisfied: pytz>=2020.1 in c:\\users\\serri\\documents\\coding\\lib\\site-packages (from pandas) (2023.3.post1)\n",
      "Requirement already satisfied: tzdata>=2022.1 in c:\\users\\serri\\documents\\coding\\lib\\site-packages (from pandas) (2023.3)\n",
      "Requirement already satisfied: tenacity>=6.2.0 in c:\\users\\serri\\documents\\coding\\lib\\site-packages (from plotly) (8.2.2)\n",
      "Requirement already satisfied: ipykernel>=4.5.1 in c:\\users\\serri\\documents\\coding\\lib\\site-packages (from ipywidgets) (6.25.0)\n",
      "Requirement already satisfied: ipython>=6.1.0 in c:\\users\\serri\\documents\\coding\\lib\\site-packages (from ipywidgets) (8.15.0)\n",
      "Requirement already satisfied: traitlets>=4.3.1 in c:\\users\\serri\\documents\\coding\\lib\\site-packages (from ipywidgets) (5.7.1)\n",
      "Requirement already satisfied: widgetsnbextension~=4.0 in c:\\users\\serri\\documents\\coding\\lib\\site-packages (from ipywidgets) (4.0.5)\n",
      "Requirement already satisfied: jupyterlab-widgets~=3.0 in c:\\users\\serri\\documents\\coding\\lib\\site-packages (from ipywidgets) (3.0.5)\n",
      "Requirement already satisfied: comm>=0.1.1 in c:\\users\\serri\\documents\\coding\\lib\\site-packages (from ipykernel>=4.5.1->ipywidgets) (0.1.2)\n",
      "Requirement already satisfied: debugpy>=1.6.5 in c:\\users\\serri\\documents\\coding\\lib\\site-packages (from ipykernel>=4.5.1->ipywidgets) (1.6.7)\n",
      "Requirement already satisfied: jupyter-client>=6.1.12 in c:\\users\\serri\\documents\\coding\\lib\\site-packages (from ipykernel>=4.5.1->ipywidgets) (7.4.9)\n",
      "Requirement already satisfied: jupyter-core!=5.0.*,>=4.12 in c:\\users\\serri\\documents\\coding\\lib\\site-packages (from ipykernel>=4.5.1->ipywidgets) (5.3.0)\n",
      "Requirement already satisfied: matplotlib-inline>=0.1 in c:\\users\\serri\\documents\\coding\\lib\\site-packages (from ipykernel>=4.5.1->ipywidgets) (0.1.6)\n",
      "Requirement already satisfied: nest-asyncio in c:\\users\\serri\\documents\\coding\\lib\\site-packages (from ipykernel>=4.5.1->ipywidgets) (1.5.6)\n",
      "Requirement already satisfied: packaging in c:\\users\\serri\\documents\\coding\\lib\\site-packages (from ipykernel>=4.5.1->ipywidgets) (23.1)\n",
      "Requirement already satisfied: psutil in c:\\users\\serri\\documents\\coding\\lib\\site-packages (from ipykernel>=4.5.1->ipywidgets) (5.9.0)\n",
      "Requirement already satisfied: pyzmq>=20 in c:\\users\\serri\\documents\\coding\\lib\\site-packages (from ipykernel>=4.5.1->ipywidgets) (24.0.1)\n",
      "Requirement already satisfied: tornado>=6.1 in c:\\users\\serri\\documents\\coding\\lib\\site-packages (from ipykernel>=4.5.1->ipywidgets) (6.3.2)\n",
      "Requirement already satisfied: backcall in c:\\users\\serri\\documents\\coding\\lib\\site-packages (from ipython>=6.1.0->ipywidgets) (0.2.0)\n",
      "Requirement already satisfied: decorator in c:\\users\\serri\\documents\\coding\\lib\\site-packages (from ipython>=6.1.0->ipywidgets) (5.1.1)\n",
      "Requirement already satisfied: jedi>=0.16 in c:\\users\\serri\\documents\\coding\\lib\\site-packages (from ipython>=6.1.0->ipywidgets) (0.18.1)\n",
      "Requirement already satisfied: pickleshare in c:\\users\\serri\\documents\\coding\\lib\\site-packages (from ipython>=6.1.0->ipywidgets) (0.7.5)\n",
      "Requirement already satisfied: prompt-toolkit!=3.0.37,<3.1.0,>=3.0.30 in c:\\users\\serri\\documents\\coding\\lib\\site-packages (from ipython>=6.1.0->ipywidgets) (3.0.36)\n",
      "Requirement already satisfied: pygments>=2.4.0 in c:\\users\\serri\\documents\\coding\\lib\\site-packages (from ipython>=6.1.0->ipywidgets) (2.15.1)\n",
      "Requirement already satisfied: stack-data in c:\\users\\serri\\documents\\coding\\lib\\site-packages (from ipython>=6.1.0->ipywidgets) (0.2.0)\n",
      "Requirement already satisfied: colorama in c:\\users\\serri\\documents\\coding\\lib\\site-packages (from ipython>=6.1.0->ipywidgets) (0.4.6)\n",
      "Requirement already satisfied: six>=1.5 in c:\\users\\serri\\documents\\coding\\lib\\site-packages (from python-dateutil>=2.8.2->pandas) (1.16.0)\n",
      "Requirement already satisfied: parso<0.9.0,>=0.8.0 in c:\\users\\serri\\documents\\coding\\lib\\site-packages (from jedi>=0.16->ipython>=6.1.0->ipywidgets) (0.8.3)\n",
      "Requirement already satisfied: entrypoints in c:\\users\\serri\\documents\\coding\\lib\\site-packages (from jupyter-client>=6.1.12->ipykernel>=4.5.1->ipywidgets) (0.4)\n",
      "Requirement already satisfied: platformdirs>=2.5 in c:\\users\\serri\\documents\\coding\\lib\\site-packages (from jupyter-core!=5.0.*,>=4.12->ipykernel>=4.5.1->ipywidgets) (3.10.0)\n",
      "Requirement already satisfied: pywin32>=300 in c:\\users\\serri\\documents\\coding\\lib\\site-packages (from jupyter-core!=5.0.*,>=4.12->ipykernel>=4.5.1->ipywidgets) (305.1)\n",
      "Requirement already satisfied: wcwidth in c:\\users\\serri\\documents\\coding\\lib\\site-packages (from prompt-toolkit!=3.0.37,<3.1.0,>=3.0.30->ipython>=6.1.0->ipywidgets) (0.2.5)\n",
      "Requirement already satisfied: executing in c:\\users\\serri\\documents\\coding\\lib\\site-packages (from stack-data->ipython>=6.1.0->ipywidgets) (0.8.3)\n",
      "Requirement already satisfied: asttokens in c:\\users\\serri\\documents\\coding\\lib\\site-packages (from stack-data->ipython>=6.1.0->ipywidgets) (2.0.5)\n",
      "Requirement already satisfied: pure-eval in c:\\users\\serri\\documents\\coding\\lib\\site-packages (from stack-data->ipython>=6.1.0->ipywidgets) (0.2.2)\n",
      "Note: you may need to restart the kernel to use updated packages.\n"
     ]
    }
   ],
   "source": [
    "pip install pandas plotly numpy ipywidgets"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "a2c48ad0-8b68-4568-9943-1566f04b2fa0",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import plotly.express as px\n",
    "import plotly.graph_objects as go\n",
    "from plotly.subplots import make_subplots\n",
    "import numpy as np\n",
    "from ipywidgets import interact, widgets\n",
    "from IPython.display import display"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "38fb4999-4626-4526-b53d-aaa278c9d600",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "🎛️ Select filters and the dashboard will update automatically:\n",
      "(Hold Ctrl/Cmd to select multiple items)\n",
      "\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "310bee9cf4924339b17fceb0a128247a",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "interactive(children=(SelectMultiple(description='Regions:', index=(0, 1, 2, 3), options=('West', 'North', 'So…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<function __main__.create_dashboard(regions, products)>"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Generate sample data\n",
    "np.random.seed(42)\n",
    "dates = pd.date_range('2024-01-01', '2024-12-31', freq='D')\n",
    "df = pd.DataFrame({\n",
    "    'Date': dates,\n",
    "    'Sales': np.random.randint(1000, 5000, size=len(dates)),\n",
    "    'Region': np.random.choice(['North', 'South', 'East', 'West'], size=len(dates)),\n",
    "    'Product': np.random.choice(['Product A', 'Product B', 'Product C'], size=len(dates))\n",
    "})\n",
    "\n",
    "# Create interactive function\n",
    "def create_dashboard(regions, products):\n",
    "    # Filter data\n",
    "    filtered_df = df[\n",
    "        (df['Region'].isin(regions)) & \n",
    "        (df['Product'].isin(products))\n",
    "    ]\n",
    "    \n",
    "    # Create subplot figure\n",
    "    fig = make_subplots(\n",
    "        rows=2, cols=2,\n",
    "        subplot_titles=('Sales Over Time', 'Sales by Region', \n",
    "                       'Sales by Product', 'Key Metrics'),\n",
    "        specs=[[{\"type\": \"scatter\"}, {\"type\": \"bar\"}],\n",
    "               [{\"type\": \"pie\"}, {\"type\": \"table\"}]]\n",
    "    )\n",
    "    \n",
    "    # 1. Line chart - Sales over time\n",
    "    fig.add_trace(\n",
    "        go.Scatter(x=filtered_df['Date'], y=filtered_df['Sales'], \n",
    "                  mode='lines', name='Daily Sales'),\n",
    "        row=1, col=1\n",
    "    )\n",
    "    \n",
    "    # 2. Bar chart - Sales by region\n",
    "    sales_by_region = filtered_df.groupby('Region')['Sales'].sum().reset_index()\n",
    "    fig.add_trace(\n",
    "        go.Bar(x=sales_by_region['Region'], y=sales_by_region['Sales'],\n",
    "              name='Sales by Region'),\n",
    "        row=1, col=2\n",
    "    )\n",
    "    \n",
    "    # 3. Pie chart - Sales by product\n",
    "    sales_by_product = filtered_df.groupby('Product')['Sales'].sum().reset_index()\n",
    "    fig.add_trace(\n",
    "        go.Pie(labels=sales_by_product['Product'], values=sales_by_product['Sales'],\n",
    "              name='Product Distribution'),\n",
    "        row=2, col=1\n",
    "    )\n",
    "    \n",
    "    # 4. Metrics table\n",
    "    metrics = pd.DataFrame({\n",
    "        'Metric': ['Total Sales', 'Avg Daily Sales', 'Max Daily Sales', 'Days'],\n",
    "        'Value': [\n",
    "            f\"${filtered_df['Sales'].sum():,}\",\n",
    "            f\"${filtered_df['Sales'].mean():.0f}\",\n",
    "            f\"${filtered_df['Sales'].max():,}\",\n",
    "            f\"{len(filtered_df)}\"\n",
    "        ]\n",
    "    })\n",
    "    \n",
    "    fig.add_trace(\n",
    "        go.Table(\n",
    "            header=dict(values=list(metrics.columns),\n",
    "                       fill_color='paleturquoise',\n",
    "                       align='left'),\n",
    "            cells=dict(values=[metrics.Metric, metrics.Value],\n",
    "                      fill_color='lavender',\n",
    "                      align='left')\n",
    "        ),\n",
    "        row=2, col=2\n",
    "    )\n",
    "    \n",
    "    # Update layout\n",
    "    fig.update_layout(\n",
    "        height=800,\n",
    "        showlegend=False,\n",
    "        title_text=\"📊 Interactive Sales Dashboard\"\n",
    "    )\n",
    "    \n",
    "    fig.show()\n",
    "    \n",
    "    # Display raw data option\n",
    "    print(f\"\\n{'='*60}\")\n",
    "    print(f\"Filtered Data: {len(filtered_df)} rows\")\n",
    "    print(f\"{'='*60}\")\n",
    "    display(filtered_df.head(10))\n",
    "\n",
    "# Create interactive widgets\n",
    "region_widget = widgets.SelectMultiple(\n",
    "    options=df['Region'].unique().tolist(),\n",
    "    value=df['Region'].unique().tolist(),\n",
    "    description='Regions:',\n",
    "    disabled=False\n",
    ")\n",
    "\n",
    "product_widget = widgets.SelectMultiple(\n",
    "    options=df['Product'].unique().tolist(),\n",
    "    value=df['Product'].unique().tolist(),\n",
    "    description='Products:',\n",
    "    disabled=False\n",
    ")\n",
    "\n",
    "# Display interactive dashboard\n",
    "print(\"🎛️ Select filters and the dashboard will update automatically:\")\n",
    "print(\"(Hold Ctrl/Cmd to select multiple items)\\n\")\n",
    "\n",
    "interact(create_dashboard, \n",
    "         regions=region_widget, \n",
    "         products=product_widget)\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
