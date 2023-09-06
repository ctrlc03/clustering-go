package kmeans

import (
	"fmt"
	"math"
	"math/rand"
	"time"
	"encoding/json"
	"os"
)

type Ballot struct {
	Votes []Vote `json:"votes"`
}

type Vote struct {
	VoteOption int	 `json:"voteOption"`
	VoteWeight float64	 `json:"voteWeight"`
}

type VotersCoefficients struct {
	VoterIndex int64
	ClusterIndex int8 
	Coefficient float64 
}

type Coefficient struct {
	ClusterIndex int8
	Coefficient float64
}

// KMeans is the main structure 
type KMeans struct {
	K int 									`json:"k"`
	Projects int							`json:"projects"`	
	Ballots []Ballot						`json:"ballots"`
	Vectors [][]float64						`json:"vectors"`
	InitialCentroids [][]float64			`json:"initial_centroids"`
	Centroids [][]float64					`json:"centroids"`
	PreviousCentroids [][]float64			`json:"previous_centroids"`
	Assignments []int8						`json:"assignments"`
	ClustersSize []int32					`json:"clusters_size"`
	Coefficients []Coefficient				`json:"coefficients"`
	VotersCoefficients []VotersCoefficients	`json:"voters_coefficients"`
	TraditionalQFs []float64				`json:"traditional_qfs"`
	Tolerance float64						`json:"tolerance"`
	HasConverged bool 						`json:"has_converged"`
	MaxIterations int						`json:"max_iterations"` 
	Voters int 								`json:"voters"`
	WCSS float64							`json:"wcss"`
	SilhoutteScore float64					`json:"silhoutte_score"`
	Distances []float64						`json:"distances"`
	DunnIndex float64 						`json:"dunn_index"`
	DaviesBouldinIndex float64				`json:"davies_bouldin_index"`
} 

// Init initializes the KMeans object
func NewKMeans(k int, ballots []Ballot, projects int, maxIterations int, tolerance float64) *KMeans{
	fmt.Sprintf("Init KMeans with k=%d", k)

	kmeans := &KMeans{}

	kmeans.K = k
	kmeans.Ballots = ballots
	kmeans.MaxIterations = maxIterations
	kmeans.Tolerance = tolerance
	kmeans.Projects = projects
	kmeans.AddZeroVotesToBallots()
	kmeans.ConvertBallotsToWeights()
	kmeans.Voters = len(kmeans.Vectors)
	return kmeans 
}

/// Train trains the KMeans object
func (kmeans *KMeans) Train() {
	kmeans.CalculateInitialCentroidsCosine()
	kmeans.CalculateTraditionalQF()

	for index := 0; index < kmeans.MaxIterations; index++ {
		kmeans.AssignVotesToClusterCosine()
		kmeans.UpdateCentroids()
		kmeans.CheckConvergenceCosine()

		if (kmeans.HasConverged) {
			break 
		}
	}

	kmeans.CalculateClustersSize()
	kmeans.AssignVotersCoefficient()	
}

// ExtractZeroVotes extracts the zero votes from the ballots
func extractZeroVotes(ballot Ballot, projects int) Ballot {
	existingVoteOptions := make(map[int]bool)
	for _, vote := range ballot.Votes {
		existingVoteOptions[vote.VoteOption] = true
	}

	var updatedVotes []Vote
	for option := 0; option < projects; option++ {
		voteWeight := 0.0
		if existingVoteOptions[option] {
			for _, vote := range ballot.Votes {
				if vote.VoteOption == option {
					voteWeight = vote.VoteWeight
					break
				}
			}
		}
		updatedVotes = append(updatedVotes, Vote{VoteOption: option, VoteWeight: voteWeight})
	}

	return Ballot{Votes: updatedVotes}
}

// AddZeroVotesToBallots adds zero votes to the ballots to have the same length
func (kmeans *KMeans) AddZeroVotesToBallots() {
	for index := range kmeans.Ballots {
		missingVotes := extractZeroVotes(kmeans.Ballots[index], kmeans.Projects)
		kmeans.Ballots[index].Votes = missingVotes.Votes // Update the original ballot's votes
	}
}

// WriteToFile writes the KMeans object to a file
func (kmeans *KMeans) WriteToFile(path string) {
	file, err := os.Create(path)
	if err != nil {
		fmt.Println("Error creating file:", err)
		return
	}
	defer file.Close()

	encoder := json.NewEncoder(file)
	// marshal the data into a JSON string and write to a file 
	err = encoder.Encode(kmeans)
	if err != nil {
		fmt.Println("Error encoding JSON:", err)
		return
	}
}

/// CalculateCosineSimilarity calculate the cosine similarity between two vectors
func CalculateCosineSimilarity(vector1 []float64, vector2 []float64) float64 {
	if vector1 == nil || vector2 == nil {
		return 0
	}

	if len(vector1) != len(vector2) {
		return 0
	}

	var dotProduct float64 = 0
	var magnitude1 float64 = 0
	var magnitude2 float64 = 0

	for i := 0; i < len(vector1); i++ {
		dotProduct += vector1[i] * vector2[i]
		magnitude1 += vector1[i] * vector1[i]
		magnitude2 += vector2[i] * vector2[i]
	}

	if magnitude1 == 0 || magnitude2 == 0 {
		return 0
	}

	return dotProduct / (math.Sqrt(magnitude1) * math.Sqrt(magnitude2))
}

// CalculateCosineDistance the cosine distance between the weights and the centroids
func CalculateCosineDistance(weights [][]float64, centroids [][]float64) []float64 {
	var distances []float64

	for _, weight := range weights {
		maxDistance := -1.0
		for _, centroid := range centroids {
			distance := 1 - CalculateCosineSimilarity(weight, centroid)
			if distance > maxDistance {
				maxDistance = distance
			}
		}

		// Append the maximum distance
		distances = append(distances, maxDistance)
	}

	return distances 
}

/// calculateRandomNumberNotIncluded calculates a random number between 0 and edge
func calculateRandomNumberNotIncluded(edge int) int {
	rand.Seed(time.Now().UnixNano())
	randomNumber := rand.Intn(int(edge))

	return randomNumber
}

/// sliceEqual compare two slices to check if they are the same
func sliceEqual(a, b []float64) bool {
	if len(a) != len(b) {
		return false
	}

	for i, v := range a {
		if v != b[i] {
			return false
		}
	}
	return true
}

/// containsSlice check if a 2D slice contains a 1D slice
func containsSlice(twoDSlice [][]float64, oneDSlice []float64) bool {
	for _, slice := range twoDSlice {
		if sliceEqual(slice, oneDSlice) {
			return true
		}
	}

	return false
}

/// sumUpSlice sums up all the values in a slice
func sumUpSlice(slice []float64) float64 {
	var sum float64 = 0
	for _, value := range slice {
		sum += value
	}

	return sum
}

// ConvertBallotsToWeights converts the ballots to a 2D slice of weights
func (kmeans *KMeans) ConvertBallotsToWeights() {
	var weights [][]float64

	for _, ballot := range kmeans.Ballots {
		var weight []float64
		for _, vote := range ballot.Votes {
			weight = append(weight, vote.VoteWeight)
		}

		weights = append(weights, weight)
	}

	kmeans.Vectors = weights
}

// FindNumberOfProjects finds the number of projects
func FindNumberOfProjects(ballots []Ballot) int {
	var largestVoteIndex int = 0
	for _, ballot := range ballots {
		for _, vote := range ballot.Votes {
			if int(vote.VoteOption) > largestVoteIndex {
				largestVoteIndex = int(vote.VoteOption)
			}
		}
	}

	return largestVoteIndex
}

/// @dev select the next centroid with a probability proportional to distance
func (kmeans KMeans) selectNextCentroid(distancesOrSimilarities []float64) []float64 {
	sum := sumUpSlice(distancesOrSimilarities)

	// calculate the probabilities
	var probabilities []float64
	for _, distance := range distancesOrSimilarities {
		probabilities = append(probabilities, distance / sum)
	}

	// generate a random number between 0 and 1
	random := rand.Float64()

	var sumOfProbabilities float64 = 0
	for index, probability := range probabilities {
		sumOfProbabilities += probability
		if random <= sumOfProbabilities {
			return kmeans.Vectors[index]
		}
	}

	// fallback (return the last one)
	return kmeans.Vectors[len(kmeans.Vectors) - 1]
}

/// CalculateInitialCentroidsCosine calculates the initial centroids using the cosine distance 
func (kmeans *KMeans) CalculateInitialCentroidsCosine() {
	var centroids [][]float64

	// select the first random 
	centroids = append(centroids, kmeans.Vectors[calculateRandomNumberNotIncluded((len(kmeans.Vectors)))])

	// while the number of centroids is less than k
	for len(centroids) < kmeans.K {
		distances := CalculateCosineDistance(kmeans.Vectors, centroids)
		// select the next centroid with a probability proportional to distance
		centroid := kmeans.selectNextCentroid(distances)

		// make sure we don't already have it in the centroids
		if !containsSlice(centroids, centroid) {
			centroids = append(centroids, centroid)
		}
	}

	// store in our object
	kmeans.InitialCentroids = centroids
	kmeans.Centroids = centroids
}

/// AssignVotesToClusterCosine assigns ballots to a cluster using the cosine distance
func (kmeans *KMeans) AssignVotesToClusterCosine() {

	var assignments []int8
	var distances []float64

	for _, vector := range kmeans.Vectors {
		minDistance := math.Inf(1)
		clusterIndex := -1

		for index, centroid := range kmeans.Centroids {
			// we can break out early if the vote is the same as the centroid
			if sliceEqual(vector, centroid) {
				clusterIndex = index
				break 
			}

			// calculate the distance 
			distance := 1 - CalculateCosineSimilarity(vector, centroid)

			// compare
			if (distance < minDistance) {
				minDistance = distance
				clusterIndex = index
			}

		}

		assignments = append(assignments, int8(clusterIndex))
		if minDistance == math.Inf(1) {
			distances = append(distances, 0)
		} else {
			distances = append(distances, minDistance) 
		}
	}



	// store 
	kmeans.Assignments = assignments
	kmeans.Distances = distances
}

// UpdateCentroids updates the centroids using the cosine distance
func (kmeans *KMeans) UpdateCentroids() {
	var newCentroids [][]float64
	// loop through our clusters
	for index:= 0; index < kmeans.K; index++ {
		// get the votes for this cluster
		var votes [][]float64
		for i, assignment := range kmeans.Assignments {
			if int8(index) == assignment {
				votes = append(votes, kmeans.Vectors[i])
			}
		}

		// a cluster might not have any votes
		if len(votes) == 0 {
			continue 
		}

		// store the tmp mean here
		tmpMean := make([]float64, len(votes[0]))

		// loop for however many projects we have
		for i := 0; i < len(votes[0]); i++ {
			var sum float64 = 0
			for _, vote := range votes {
				sum += vote[i]
			}

			tmpMean[i] = sum / float64(len(votes))
		}

		newCentroids = append(newCentroids, tmpMean)
	}  

	// store the previous centroids so we can check for convergence
	kmeans.PreviousCentroids = kmeans.Centroids
	// store the updated centroids
	kmeans.Centroids = newCentroids
}

// CalculateClustersSize calculates the size of each cluster
func (kmeans *KMeans) CalculateClustersSize() {
	clusterSizes := make([]int32, len(kmeans.Centroids))

	for _, assignment := range kmeans.Assignments {
		clusterSizes[assignment]++
	}

	kmeans.ClustersSize = clusterSizes
}

// CalculateCoefficientsSmallerGroups calculates the coefficients with 
// the formula to reward smaller groups
func (kmeans *KMeans) CalculateCoefficientsSmallerGroups() {
	var coefficients []Coefficient

	// loop through all clusters
	for index, clusterSize := range kmeans.ClustersSize {
		var coefficient float64 
		if clusterSize != 0 {
			coefficient = 1 - float64(clusterSize) / float64(len(kmeans.Vectors))
		} else {
			coefficient  = 0
		}

		coefficients = append(coefficients, Coefficient{ClusterIndex: int8(index), Coefficient: coefficient})
	}

	kmeans.Coefficients = coefficients
}

// AssignVotersCoefficient assigns each user to their cluster and coefficient
func (kmeans *KMeans) AssignVotersCoefficient() {
	var votersCoefficients []VotersCoefficients

	for index, assignment := range kmeans.Assignments {
		var coeff float64 = 1 // default value

		// Find the corresponding coefficient
		for _, c := range kmeans.Coefficients {
			if c.ClusterIndex == assignment {
				coeff = c.Coefficient
				break
			}
		}

		votersCoefficients = append(votersCoefficients, VotersCoefficients{
			VoterIndex: int64(index),
			ClusterIndex: assignment,
			Coefficient: coeff,
		})
	}

	kmeans.VotersCoefficients = votersCoefficients
}

// CalculateTraditionalQF calculates the traditional QF
func (kmeans *KMeans) CalculateTraditionalQF() {
	traditionalQFs := make([]float64, kmeans.Projects)

	for projectIndex := 0; projectIndex < kmeans.Projects; projectIndex++ {
		var sum float64 = 0
		for index := 0; index < len(kmeans.Vectors); index++ {
			sum += math.Sqrt(kmeans.Vectors[index][projectIndex])
		}
		traditionalQFs = append(traditionalQFs, math.Pow(sum, 2))
	}

	kmeans.TraditionalQFs = traditionalQFs
}

// CheckConvergenceCosine checks if the centroids have converged based on cosine similarity
func (kmeans *KMeans) CheckConvergenceCosine() {
	// by default we have converged
	kmeans.HasConverged = true  

	if len(kmeans.PreviousCentroids) != len(kmeans.Centroids) || len(kmeans.PreviousCentroids[0]) != len(kmeans.Centroids[0]) {
		kmeans.HasConverged = false 
		return 
	}


	// loop through all centroids 
	for index, centroid := range kmeans.Centroids {
		distance := 1 - CalculateCosineSimilarity(centroid, kmeans.PreviousCentroids[index])
		if distance > kmeans.Tolerance {
			kmeans.HasConverged = false 
			return 
		}
	}
}

// CalculateWCSS calculates the Within-Cluster Sum of Squares (WCSS) score for KMeans clustering
func (kmeans *KMeans) CalculateWCSS() {
	wcss := 0.0
	for i := range kmeans.Vectors {
		distance := kmeans.Distances[i]
		wcss += math.Pow(distance, 2)
	}
	
	kmeans.WCSS = wcss 
}

// CalculateSilhouetteScore calculates the silhouette score for the clusters
func (kmeans *KMeans) CalculateSilhouetteScore() {
	var silhouetteSum float64
	for i, vector := range kmeans.Vectors {
		a := kmeans.calculateAverageDistanceToCluster(vector, kmeans.Assignments[i], kmeans.Vectors, kmeans.Centroids)
		b := kmeans.calculateMinAverageDistanceToOtherClusters(vector, kmeans.Assignments[i], kmeans.Vectors, kmeans.Centroids)
		// Handle NaN or zero values
		if !math.IsNaN(a) && !math.IsNaN(b) && a != 0 && b != 0 {
			silhouette := (b - a) / math.Max(a, b)
			silhouetteSum += silhouette
		}
	}

	kmeans.SilhoutteScore = silhouetteSum / float64(len(kmeans.Vectors))
}

// calculateAverageDistanceToCluster calculates the average distance of a point to other points in the same cluster
func (kmeans KMeans) calculateAverageDistanceToCluster(vector []float64, clusterIndex int8, vectors [][]float64, centroids [][]float64) float64 {
	var sumDistance float64
	count := 0

	for i := range vectors {
		if kmeans.Assignments[i] == clusterIndex {
			distance := 1 - CalculateCosineSimilarity(vector, vectors[i])
			if !math.IsNaN(distance) {
				sumDistance += distance
				count++
			}
		}
	}

	if count == 0 {
		return 0 // Return 0 to avoid division by zero
	}

	return sumDistance / float64(count)
}

// calculateMinAverageDistanceToOtherClusters calculates the minimum average distance of a point to points in other clusters
func (kmeans KMeans) calculateMinAverageDistanceToOtherClusters(vector []float64, clusterIndex int8, vectors [][]float64, centroids [][]float64) float64 {
	minDistance := math.Inf(1)

	for index := range centroids {
		if int8(index) != clusterIndex {
			averageDistance := kmeans.calculateAverageDistanceToCluster(vector, int8(index), vectors, centroids)
			if !math.IsNaN(averageDistance) && averageDistance < minDistance {
				minDistance = averageDistance
			}
		}
	}

	return minDistance
}

// CalculateDunnIndex calculates the Dunn index for the clusters
func (kmeans *KMeans) CalculateDunnIndex() {
	var minInterClusterDist = math.Inf(1)
	var maxIntraClusterDist float64 = 0

	// Calculate minimum inter-cluster distance
	for i := 0; i < len(kmeans.Centroids); i++ {
		for j := i + 1; j < len(kmeans.Centroids); j++ {
			dist := 1 - CalculateCosineSimilarity(kmeans.Centroids[i], kmeans.Centroids[j])
			if dist < minInterClusterDist {
				minInterClusterDist = dist
			}
		}
	}

	// Calculate maximum intra-cluster distance
	for i, vector := range kmeans.Vectors {
		for j := i + 1; j < len(kmeans.Vectors); j++ {
			if kmeans.Assignments[i] == kmeans.Assignments[j] {
				dist := 1 - CalculateCosineSimilarity(vector, kmeans.Vectors[j])
				if dist > maxIntraClusterDist {
					maxIntraClusterDist = dist
				}
			}
		}
	}

	// Calculate Dunn index
	if maxIntraClusterDist == 0 {
		kmeans.DunnIndex = 0
	} else {
		kmeans.DunnIndex = minInterClusterDist / maxIntraClusterDist
	}
}

// CalculateAverageDistances calculates the average distance of each point in a cluster to its centroid
func (kmeans *KMeans) CalculateAverageDistances() []float64 {
	averageDistances := make([]float64, len(kmeans.Centroids))

	for i, centroid := range kmeans.Centroids {
		var sumDistances float64
		var count int

		for j, assignment := range kmeans.Assignments {
			if int8(i) == assignment {
				distance := 1 - CalculateCosineSimilarity(kmeans.Vectors[j], centroid)
				sumDistances += distance
				count++
			}
		}

		if count > 0 {
			averageDistances[i] = sumDistances / float64(count)
		}
	}

	return averageDistances
}

// CalculateDaviesBouldinIndex calculates the Davies-Bouldin index for the clusters
func (kmeans *KMeans) CalculateDaviesBouldinIndex() {
	averageDistances := kmeans.CalculateAverageDistances()
	var sumRatios float64

	for i := range kmeans.Centroids {
		var maxRatio float64

		for j := range kmeans.Centroids {
			if i != j {
				distanceBetweenCentroids := 1 - CalculateCosineSimilarity(kmeans.Centroids[i], kmeans.Centroids[j])
				ratio := (averageDistances[i] + averageDistances[j]) / distanceBetweenCentroids

				if ratio > maxRatio {
					maxRatio = ratio
				}
			}
		}

		sumRatios += maxRatio
	}

	kmeans.DaviesBouldinIndex = sumRatios / float64(len(kmeans.Centroids))
}