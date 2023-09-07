package main

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"
	"strconv"
	"sync"

	"github.com/ctrlc03/clustering-go/src/kmeans"
	"github.com/ctrlc03/clustering-go/src/plotting"
)

type Vote struct {
	VoteOption string	 `json:"voteOption"`
	VoteWeight string	 `json:"voteWeight"`
}

// parseData parses the data from the JSON file and returns a slice of Ballot objects
func parseData(path string) []kmeans.Ballot {
	jsonFile, err := os.Open(path)
	if err != nil {
		panic(err)
	}
	defer jsonFile.Close()

	byteValue, _ := ioutil.ReadAll(jsonFile)

	var ballots []map[string]Vote
	if err := json.Unmarshal(byteValue, &ballots); err != nil {
		panic(err)
	}


	var convertedBallots []kmeans.Ballot
	for _, ballot := range ballots {
		var votes []kmeans.Vote
		for _, vote := range ballot {
			voteOption, _ := strconv.Atoi(vote.VoteOption)
			voteWeight, _ := strconv.ParseFloat(vote.VoteWeight, 64)
			votes = append(votes, kmeans.Vote{VoteOption: voteOption, VoteWeight: voteWeight})
		}
		convertedBallots = append(convertedBallots, kmeans.Ballot{Votes: votes})
	}

	return convertedBallots

}

func main() {
	folder := "./data/"
	outputFolderPlot := "./data/output/plots/"
	outputFolderJson := "./data/output/json/"
	files, err := listFilesInFolder(folder)
	if err != nil {
		panic(err)
	}

	var wg sync.WaitGroup

	for _, file := range files {
		wg.Add(1)
		fmt.Println("Processing file", file)

		go func(filePath string) {
			defer wg.Done()
			ballots := parseData(folder + filePath)
			numOfProjects := kmeans.FindNumberOfProjects(ballots)

			for iteration := 0; iteration < 1; iteration++ {
				wg.Add(1)
				go func(iter int) {
					defer wg.Done()
					var wcss []float64
					var silouhtte []float64
					var dunnIndex []float64
					var daviesBouldinIndex []float64
					for i := 3; i < 20; i++ {
						kmeansObj := kmeans.NewKMeans(i, ballots, numOfProjects, 100, 0.001)
						kmeansObj.Train()
						wcss = append(wcss, kmeansObj.WCSS)
						silouhtte = append(silouhtte, kmeansObj.SilhoutteScore)
						dunnIndex = append(dunnIndex, kmeansObj.DunnIndex)
						daviesBouldinIndex = append(daviesBouldinIndex, kmeansObj.DaviesBouldinIndex)

						// Save the KMeans object to a JSON file
						kmeansObj.WriteToFile(outputFolderJson + filePath + "_k_" + strconv.Itoa(i) + "_iteration_" + strconv.Itoa(iter) + ".json")
					}

					// Plot the results
					plotting.PlotByWCSS(wcss, outputFolderPlot + filePath + "_elbow_method_" + strconv.Itoa(iter) + ".png")
					plotting.PlotByFeature(silouhtte, outputFolderPlot + filePath + "_silhouette_score_" + strconv.Itoa(iter) + ".png", "Silhouette Score")
					plotting.PlotByFeature(dunnIndex, outputFolderPlot + filePath + "_dunn_index_" + strconv.Itoa(iter) + ".png", "Dunn Index")
					plotting.PlotByFeature(daviesBouldinIndex, outputFolderPlot + filePath + "_davies_bouldin_index_" + strconv.Itoa(iter) + ".png", "Davies Bouldin Index")

					fmt.Println("Completed file", filePath, "iteration", iter)
				}(iteration)
			}
		}(file)
	}

	wg.Wait()
}

func listFilesInFolder(folderPath string) ([]string, error) {
	var files []string

	// Read the directory contents
	dirEntries, err := ioutil.ReadDir(folderPath)
	if err != nil {
		return nil, err
	}

	// Filter out files from the entries
	for _, entry := range dirEntries {
		if entry.Mode().IsRegular() {
			files = append(files, entry.Name())
		}
	}

	return files, nil
}