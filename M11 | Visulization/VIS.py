import plotly.graph_objects as go
from plotly.subplots import make_subplots
import mplfinance as mpf
import seaborn as sns
import hvplot.pandas
import altair as alt
from bokeh.plotting import figure, show
from bokeh.layouts import column
from bokeh.models import HoverTool
import matplotlib.pyplot as plt

class FinancialVisualizer:
    def __init__(self, df):
        """Initialize with a pandas DataFrame containing financial data"""
        self.df = df
        # Set style for static plots
        sns.set_style("darkgrid")
        
    def plot_candlestick(self, engine='plotly', show_volume=True):
        """
        Create candlestick chart using specified engine
        Engines: 'plotly', 'mplfinance'
        """
        if engine == 'plotly':
            fig = make_subplots(rows=2 if show_volume else 1, cols=1, 
                              shared_xaxes=True,
                              vertical_spacing=0.03,
                              row_heights=[0.7, 0.3] if show_volume else [1])

            fig.add_trace(go.Candlestick(x=self.df.index,
                                       open=self.df['Open'],
                                       high=self.df['High'],
                                       low=self.df['Low'],
                                       close=self.df['Close']))
            
            if show_volume:
                fig.add_trace(go.Bar(x=self.df.index, 
                                   y=self.df['Volume']),
                             row=2, col=1)

            fig.update_layout(title='Asset Price',
                            yaxis_title='Price ($)',
                            xaxis_rangeslider_visible=False)
            return fig

        elif engine == 'mplfinance':
            return mpf.plot(self.df, type='candle', 
                          volume=show_volume,
                          style='charles',
                          title='Asset Price',
                          ylabel='Price ($)')

    def plot_technical_analysis(self):
        """Create comprehensive technical analysis dashboard"""
        fig = make_subplots(rows=2, cols=2,
                           subplot_titles=('Price & Volume', 'Returns Distribution',
                                         'Correlation Heatmap', 'Moving Averages'))

        # Price and Volume
        fig.add_trace(go.Candlestick(x=self.df.index,
                                    open=self.df['Open'],
                                    high=self.df['High'],
                                    low=self.df['Low'],
                                    close=self.df['Close']),
                     row=1, col=1)

        # Returns Distribution
        fig.add_trace(go.Histogram(x=self.df['Returns'],
                                 nbinsx=50,
                                 name='Returns Distribution'),
                     row=1, col=2)

        # Correlation Heatmap
        corr = self.df[['Open', 'High', 'Low', 'Close', 'Volume']].corr()
        fig.add_trace(go.Heatmap(z=corr.values,
                                x=corr.index,
                                y=corr.columns,
                                colorscale='RdBu'),
                     row=2, col=1)

        # Moving Averages
        fig.add_trace(go.Scatter(x=self.df.index,
                               y=self.df['Close'].rolling(20).mean(),
                               name='MA20'),
                     row=2, col=2)
        
        fig.update_layout(height=800, title_text="Technical Analysis Dashboard")
        return fig

    def plot_interactive_analysis(self, engine='hvplot'):
        """
        Create interactive analysis plot using specified engine
        Engines: 'hvplot', 'bokeh', 'altair'
        """
        if engine == 'hvplot':
            return self.df.hvplot.line(
                x='Date', 
                y=['Close', 'MA20', 'MA50'],
                title='Interactive Price Analysis',
                height=400,
                width=800
            ).opts(
                legend_position='top_right',
                yformatter='%.0f',
                toolbar='above'
            )

        elif engine == 'bokeh':
            p = figure(width=800, height=400, x_axis_type="datetime")
            p.line(self.df.index, self.df.Close, line_width=2)
            p.circle(self.df.index, self.df.Close, size=1)
            
            p.add_tools(HoverTool(
                tooltips=[
                    ('Date', '@x{%F}'),
                    ('Price', '$@y{0.00}')
                ],
                formatters={'@x': 'datetime'}
            ))
            return p

        elif engine == 'altair':
            return alt.Chart(self.df.reset_index()).mark_line().encode(
                x='Date:T',
                y='Close:Q',
                tooltip=['Date', 'Close']
            ).interactive()

    def plot_statistical_analysis(self):
        """Create statistical analysis plots using Seaborn"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Returns distribution
        sns.histplot(data=self.df, x='Returns', kde=True, ax=ax1)
        ax1.set_title('Returns Distribution')
        
        # Correlation heatmap
        sns.heatmap(self.df.corr(), annot=True, cmap='coolwarm', ax=ax2)
        ax2.set_title('Correlation Heatmap')
        
        return fig

# Example usage:
"""
# Load your data
df = pd.read_csv('financial_data.csv', index_col='Date', parse_dates=True)

# Create visualizer
viz = FinancialVisualizer(df)

# Create different visualizations
candlestick = viz.plot_candlestick(engine='plotly')
technical = viz.plot_technical_analysis()
interactive = viz.plot_interactive_analysis(engine='hvplot')
statistical = viz.plot_statistical_analysis()

# Show plots
candlestick.show()
technical.show()
interactive.show()
plt.show()  # For statistical analysis
""" 

