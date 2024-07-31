from io import BytesIO
import time
import requests
from PIL import Image
import panel as pn
from plotly.subplots import make_subplots
import plotly.graph_objects as go

ACCENT = "teal"

styles = {
    "box-shadow": "rgba(50, 50, 93, 0.25) 0px 6px 12px -2px, rgba(0, 0, 0, 0.3) 0px 3px 7px -3px",
    "border-radius": "4px",
    "padding": "10px",
}

image = pn.pane.JPG("https://assets.holoviz.org/panel/tutorials/wind_turbines_sunset.png")

pn.extension("plotly")
pn.extension(design="material", sizing_mode="stretch_width")
pn.config.theme = 'dark'

@pn.cache
def get_pil_image(url):
    response = requests.get(url)
    return Image.open(BytesIO(response.content))

@pn.cache
def get_plot(label):
    data = pd.Series(label).sort_values()
    return data.hvplot.barh(title="Prediction", ylim=(0, 100)).opts(default_tools=[])

def get_image_button(url, image_pane, row):
    button = pn.widgets.Button(
        width=100,
        height=100,
        stylesheets=[
            f"button {{background-image: url({url});background-size: cover;}}"
        ],
    )
    pn.bind(handle_example_click, button, url, image_pane, row, watch=True)
    return button

def handle_example_click(event, url, image_pane, row):
    f = image_pane.object
    img = get_pil_image(url)
    fig1 = go.Image(z=img)
    f.add_trace(fig1, row, 1)

def add_image(value, image_pane, row, col, txt):
    f = image_pane.object
    img = Image.open(BytesIO(value))

    # Get the image dimensions
    width, height = img.size
    fig1 = go.Image(z=img)

    # Add the image trace
    f.add_trace(fig1, row, col)

    # Default window dimension to view (adjust as needed)
    window_width = min(width, 1200)
    window_height = min(height, 400)

    # Update layout to include margins that fit the window
    f.update_layout(
        width=window_width,
        height=window_height * 2 + 100,
        margin=dict(l=0, r=0, b=0, t=30)
    )

    # Set the axes range to the window dimensions to show a part of the image
    f.update_xaxes(range=[0, window_width], row=row, col=col)
    f.update_yaxes(range=[height, height - window_height], row=row, col=col)  # Note the inversion of y-axis

def update_indicator(event, image_pane):
    if not event:
        return
    image_pane.object.data = []

def image_classification_interface(examples):
    image_view = pn.pane.Plotly(config={'displayModeBar': True})

    file_input = pn.widgets.FileInput(accept=".png,.jpeg,.jpg")
    file_input2 = pn.widgets.FileInput(accept=".png,.jpeg,.jpg")
    file_input_component = pn.Column("### Upload Image 1", file_input)
    file_input_component2 = pn.Column("### Upload Image 2", file_input2)

    examples_input_component = pn.Column(
        "### Examples image 1", pn.Row(*(get_image_button(url, image_view, 1) for url in examples))
    )
    examples_input_component2 = pn.Column(
        "### Examples image 2", pn.Row(*(get_image_button(url, image_view, 2) for url in examples))
    )

    # Create subplots with minimal vertical spacing
    f = make_subplots(rows=2, cols=1, subplot_titles=[" ", " "], vertical_spacing=0.05)
    f.update_layout(title_text="Image comparison", height=1000, width=1000)

    f.layout.annotations[0].update(text='Image 1')
    f.layout.annotations[1].update(text='Image 2')

    f.update_xaxes(matches='x')
    f.update_yaxes(matches='y')
    f.update_layout(
        paper_bgcolor="LightSteelBlue",
        dragmode='pan',  # Enable panning
        autosize=False
    )

    image_view.object = f
    pn.bind(add_image, file_input, image_view, 1, 1, 'img1', watch=True)
    pn.bind(add_image, file_input2, image_view, 2, 1, 'img2', watch=True)

    button = pn.widgets.Button(name='Remove all images', button_type='primary')
    pn.bind(update_indicator, button, image_view, watch=True)

    input_component1 = pn.Column(
        "# Display",
        image_view,
        margin=10, scroll=True
    )
    input_component2 = pn.Column(
        "# Input",
        file_input_component, file_input_component2, examples_input_component, examples_input_component2, button,
        margin=10, scroll=True, width =400
    )
    input_component3 = pn.Row(input_component1, input_component2)

    return input_component3

EXAMPLES = [
    "https://assets.holoviz.org/panel/tutorials/wind_turbine.png",
    "https://assets.holoviz.org/panel/tutorials/solar_panel.png",
    "https://assets.holoviz.org/panel/tutorials/battery_storage.png",
]

image_classification_interface(examples=EXAMPLES).servable()
