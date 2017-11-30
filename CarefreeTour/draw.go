package plot

import (
	"image/color/palette"
	"math/rand"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"image/color"
)

func NewPlot(name, x, y string) *plot.Plot {
	p, err := plot.New()
	if err != nil {
		panic(err)
	}
	p.Title.Text = name
	p.X.Label.Text = x
	p.Y.Label.Text = y

	p.X.Min = 0
	p.X.Max = 0
	p.Y.Min = 0
	p.Y.Max = 0

	return p
}

func Draw2DFuncAuto(p *plot.Plot, f func(float64) float64, name string) {
	index := rand.Intn(216) - 1
	c := palette.WebSafe[index]
	quad := plotter.NewFunction(f)
	quad.Color = c
	p.Add(quad)
	p.Legend.Add(name, quad)
}

func Draw2DPointAuto(p *plot.Plot, f [][]float64, name string) {
	index := rand.Intn(216) - 1
	c := palette.WebSafe[index]
	n := len(f)
	pts := make(plotter.XYs, n)
	for i := range pts {
		pts[i].X = f[i][0]
		pts[i].Y = f[i][1]
	}
	s, err := plotter.NewScatter(pts)
	//bs, err := plotter.NewBubbles(bubbleData, vg.Points(1), vg.Points(20))
	if err != nil {
		panic(err)
	}
	s.GlyphStyle.Color = c
	p.Add(s)
	p.Legend.Add(name, s)
	xi, xa, yi, ya := plotter.XYRange(s.XYs)
	if xi < p.X.Min {
		p.X.Min = xi
	}
	if xa > p.X.Max {
		p.X.Max = xa*1.3
	}
	if yi < p.Y.Min {
		p.Y.Min = yi
	}
	if ya > p.Y.Max {
		p.Y.Max = ya
	}
}

func Draw2DFunc(p *plot.Plot, f func(float64) float64, c color.Color, name string) {
	quad := plotter.NewFunction(f)
	quad.Color = c
	p.Add(quad)
	p.Legend.Add(name, quad)
}

func Paint(p *plot.Plot, xi, xa, yi, ya float64, name string) {
	if xi < p.X.Min {
		p.X.Min = xi
	}
	if xa > p.X.Max {
		p.X.Max = xa*1.3
	}
	if yi < p.Y.Min {
		p.Y.Min = yi
	}
	if ya > p.Y.Max {
		p.Y.Max = ya
	}
	if err := p.Save(4*vg.Inch, 4*vg.Inch, name); err != nil {
		panic(err)
	}
}
