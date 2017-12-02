package cost

import (
	"math"
)

//批量梯度下降
func BatchGradientDescent(theta []float64,bigdata [][]float64,rate float64,eps float64,max int) [][]float64{
	var count int=0
	var step [][]float64
	step=append(step,theta)
	l:=len(theta)
	for count<max {
		next:=GradientDescent(step[ len(step)-1] ,bigdata,rate)
		//fmt.Println(next)
		step=append(step,next)
		count++
		c :=true
		for i:=0;i<l ; i++ {
			r:=step[ len(step)-2][i]
			if math.Abs(r-next[i])<eps{
				c=true&&c
			}else {
				c=false&&c
			}
		}
		if c {
			break
		}
	}
	return step
}
//单次梯度下降
func GradientDescent(theta []float64,bigdata [][]float64,rate float64) []float64{
	len:=len(theta)
	next := make([]float64,len)
	for i:=0;i<len;i++ {
		next[i]=theta[i]-rate*Gradient(i,theta,bigdata)
	}
	return next
}
//梯度运算
func Gradient(index int,theta []float64,bigdata [][]float64) float64{
	var sum float64
	for _,i:=range bigdata{
		sum=sum+Node(index,theta,i,i[len(i)-1])
	}
	sum=sum/float64(len(bigdata))
	return sum
}
//节点计算
func Node(index int,theta []float64,x []float64,y float64) float64{
	var ret float64
	var xi float64
	for i:=0;i<len(theta);i++ {
		ret=ret+theta[i]*x[i]
		if index==i {
			xi=theta[i]
		}
	}
	ret=ret-y
	ret=ret*xi
	return ret
}
