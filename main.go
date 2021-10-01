package main

import (
	"bufio"
	"bytes"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"net/http"
	"os"
	"sort"

	tensorflow "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
)

const (
	graphFile  = "/model/tensorflow_inception_graph.pb"
	labelsFile = "/model/imagenet_comp_graph_label_strings.txt"
)

// Label type
type Label struct {
	Label       string  `json:"label"`
	Probability float32 `json:"probability"`
}

// Labels type
type Labels []Label

func (a Labels) Len() int           { return len(a) }
func (a Labels) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
func (a Labels) Less(i, j int) bool { return a[i].Probability > a[j].Probability }

func main() {
	os.Setenv("TF_CPP_MIN_LOG_LEVEL", "2")

	if len(os.Args) < 2 {
		log.Fatalf("usage: imgrecognition <image_url>")
	}
	fmt.Printf("url: %s\n", os.Args[1])

	// Get image from URL
	response, e := http.Get(os.Args[1])
	if e != nil {
		log.Fatalf("unable to get image from url: %v", e)
	}
	defer response.Body.Close()

	// unable to load graph and labels
	modelGraph, labels, err := loadModel()
	if err != nil {
		log.Fatalf("unable to load model: %v", err)
	}

	// Get normalized tensor     get normalized tensor ready
	tensor, err := normalizeImage(response.Body)
	if err != nil {
		log.Fatalf("unable to make a tensor from image: %v", err)
	}

	// Create a session for inference over modelGraph   get session ready
	session, err := tensorflow.NewSession(modelGraph, nil)
	if err != nil {
		log.Fatalf("could not init session: %v", err)
	}

	//defer session.Close()

	output, err := session.Run(
		map[tensorflow.Output]*tensorflow.Tensor{
			modelGraph.Operation("input").Output(0): tensor, // use input from modelGraph
		},
		[]tensorflow.Output{
			modelGraph.Operation("output").Output(0),
		},
		nil)
	if err != nil {
		log.Fatalf("could not run inference: %v", err) // unable to inference
	}

	// list top the probabilities and assign labels
	res := getTopFiveLabels(labels, output[0].Value().([][]float32)[0])
	for _, l := range res {
		fmt.Printf("label: %s, probability: %.2f%%\n", l.Label, l.Probability*100) // show percentage
	}
}

func loadModel() (*tensorflow.Graph, []string, error) {
	// Load inception model
	model, err := ioutil.ReadFile(graphFile)
	if err != nil {
		return nil, nil, err
	}
	// Load Graph
	graph := tensorflow.NewGraph()
	if err := graph.Import(model, ""); err != nil {
		return nil, nil, err
	}

	// Load labels
	labelsFile, err := os.Open(labelsFile)
	if err != nil {
		return nil, nil, err
	}
	defer labelsFile.Close()
	scanner := bufio.NewScanner(labelsFile)
	var labels []string
	for scanner.Scan() {
		labels = append(labels, scanner.Text())
	}

	return graph, labels, scanner.Err()
}

func getTopFiveLabels(labels []string, probabilities []float32) []Label {
	var resultLabels []Label
	for i, p := range probabilities {
		if i >= len(labels) {
			break // break condition
		}
		resultLabels = append(resultLabels, Label{Label: labels[i], Probability: p})
	}
	// create
	// type Label struct{
	// 	Label      string
	//      Proability float32
	// }
	// func (l Labels) Len() int {return len(l)}
	// func (l Labels) Swap() int {return len(l)}
	// func (l Labels) Less() bool {l[i].Probability < l[j].Probability }

	sort.Sort(Labels(resultLabels))
	return resultLabels[:5]
}

func normalizeImage(body io.ReadCloser) (*tensorflow.Tensor, error) {
	var buf bytes.Buffer
	io.Copy(&buf, body)

	tensor, err := tensorflow.NewTensor(buf.String())
	if err != nil {
		return nil, err
	}

	graph, input, output, err := getNormalizedGraph()
	if err != nil {
		return nil, err
	}

	// set graph
	session, err := tensorflow.NewSession(graph, nil)
	if err != nil {
		return nil, err
	}
	// normalized, err
	// normalized: a slice of tensors
	normalized, err := session.Run(
		map[tensorflow.Output]*tensorflow.Tensor{
			input: tensor,
		},
		[]tensorflow.Output{
			output,
		},
		nil)
	if err != nil {
		return nil, err
	}

	// normalized image, and return it to the main func?
	return normalized[0], nil
}

// NORMALIZE IMG to a specific Inception format: 1. normalize, resize, set constraints
// Creates a graph to decode, rezise and normalize an image
func getNormalizedGraph() (graph *tensorflow.Graph, input, output tensorflow.Output, err error) {
	s := op.NewScope()
	input = op.Placeholder(s, tensorflow.String)
	// 3 return RGB image
	decode := op.DecodeJpeg(s, input, op.DecodeJpegChannels(3))

	// Sub: returns x - y element-wise
	output = op.Sub(s,
		// make it 224x224: inception specific
		op.ResizeBilinear(s,
			// inserts a dimension of 1 into a tensor's shape.
			op.ExpandDims(s,
				// cast image to float type
				op.Cast(s, decode, tensorflow.Float),
				op.Const(s.SubScope("make_batch"), int32(0))),
			op.Const(s.SubScope("size"), []int32{224, 224})),
		// mean = 117: inception specific
		op.Const(s.SubScope("mean"), float32(117)))
	graph, err = s.Finalize()

	return graph, input, output, err
}
