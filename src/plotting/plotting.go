package plotting 

import (
	"fmt"
	"log"
	"math"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"image/color"
)

func PlotByWCSS(wcss []float64, output string) {
	p := plot.New()
	p.X.Padding = vg.Length(20)
	p.Y.Padding = vg.Length(20)
	p.Title.Text = "Elbow Method"

	points := make(plotter.XYs, len(wcss))
	for i, size := range wcss {
		points[i].X = float64(i+3)
		points[i].Y = size
	}

	line, err := plotter.NewLine(points)
	if err != nil {
		log.Fatalf("Error creating line plot: %v", err)
	}

	line.Color = color.RGBA{R: 0, G: 0, B: 255, A: 255}

	p.Add(line)


	p.X.Tick.Marker = plot.TickerFunc(func(min, max float64) []plot.Tick {
		ticks := []plot.Tick{}
		for i := math.Ceil(min); i <= max; i++ {
			ticks = append(ticks, plot.Tick{Value: i, Label: fmt.Sprintf("%.0f", i)})
		}
		return ticks
	})

	if err := p.Save(10*vg.Inch, 4*vg.Inch, output); err != nil {
		log.Fatalf("Error saving plot: %v", err)
	}

	fmt.Println("Plot saved to", output)
}

func PlotByFeature(sizes []float64, output string, title string) {
	p := plot.New()
	p.X.Padding = vg.Length(20)
	p.Y.Padding = vg.Length(20)
	p.Title.Text = title 

	bars := make(plotter.Values, len(sizes))
	labels := make([]string, len(sizes))

	for index, size := range sizes {
		bars[index] = float64(size) 
		labels[index] = fmt.Sprintf("%d", index+3) 
	}	


	barchart, err := plotter.NewBarChart(bars, vg.Points(10))

	if err != nil {
		log.Fatalf("Error creating bar chart: %v", err)
	}

	barchart.Color = color.RGBA{R: 0, G: 0, B: 255, A: 255}

	p.NominalX(labels...)
	p.Add(barchart)
	

	// Save the plot to a file
	if err := p.Save(10*vg.Inch, 4*vg.Inch, output); err != nil {
		log.Fatalf("Error saving plot: %v", err)
	}

	fmt.Println("Plot saved to ", output)
} 

