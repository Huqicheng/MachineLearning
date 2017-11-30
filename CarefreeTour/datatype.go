package datatype

import (
	"os"
	"fmt"
	"encoding/csv"
	"strconv"
	"gonum.org/v1/gonum/mat"
)

func CVS2Dense(path string) (int,int,*mat.Dense,error) {
	r,c,data,err:=CVS2Float64Array(path)
	var all []float64
	for _, i := range data {
		for _, j := range i {
			all = append(all,j)
		}
	}
	a := mat.NewDense(r, c, all)
	return r,c,a,err
}

func CVS2Float64Array (path string) (int,int,[][]float64,error) {
	var r int
	var c int
	ret:=[][]float64{}
	file, err := os.Open(path)
	if err != nil {
		fmt.Println("Error:", err)
		return r,c,nil,nil
	}
	defer file.Close()
	reader := csv.NewReader(file)
	rc,_:=reader.ReadAll()
	r=len(rc)
	c=reader.FieldsPerRecord
	for _, r := range rc {
		var row []float64
		for _, it := range r {
			item, _:= strconv.ParseFloat(it, 64)
			row = append(row,item)
		}
		ret = append(ret,row)
	}
	return r,c,ret,err
}

func Dense2Float64Array(src mat.Dense) (int,int,[][]float64,error) {
	r,c:=src.Caps()
	ret:=[][]float64{}
	for i := 0; i < r; i++ {
		var row []float64
		for j := 0; j < c; j++ {
			row = append(row,src.At(i, j))
		}
		ret = append(ret,row)
	}
	return r,c,ret,nil
}

func Float64Array2Dense(src [][]float64) (int,int,*mat.Dense,error) {
	r:=len(src)
	c:=len(src[0])
	var all []float64
	for _, i := range src {
		for _, j := range i {
			all = append(all,j)
		}
	}
	a := mat.NewDense(r, c, all)
	return r,c,a,nil
}
func Float64ArraySer(src [][]float64) (int,int,[]float64,error) {
	r:=len(src)
	c:=len(src[0])
	var all []float64
	for _, i := range src {
		for _, j := range i {
			all = append(all,j)
		}
	}
	return r,c,all,nil
}

func DenseSer(src mat.Dense) (int,int,[]float64,error) {
	r,c:=src.Caps()
	var all []float64
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			all = append(all,src.At(i, j))
		}
	}
	return r,c,all,nil
}
