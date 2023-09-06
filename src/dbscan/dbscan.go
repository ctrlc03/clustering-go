package dbscan

import (
	"github.com/ctrlc03/clustering-go/src/kmeans"
)

type DBScan struct {
	EPS float64 		`json:"eps"`
	MinPoints int 		`json:"minPoints"`
	Ballots []Ballot    `json:"ballots"`
	Assignments []int   `json:"assignments"`

}

type Ballot struct {
	Votes []Vote `json:"votes"`
	Cluster int  `json:"cluster"`
}

type Vote struct {
	VoteOption int	 	 `json:"voteOption"`
	VoteWeight float64	 `json:"voteWeight"`
}

// NewDBSCAN returns a new DBScan instance
func NewDBSCAN(eps float64, minPoints int) *DBScan {
	return &DBScan{
		EPS: eps,
		MinPoints: minPoints,
	}
}

// // Train trains the DBScan model
// func (dbscan *DBScan) Train() {
// 	cluster := 0

// 	for i := range dbscan.Ballots {
// 		if dbscan.Assignments[i] != 0 {
// 			continue 
// 		}

// 		nieghbors := dbscan.FindNeighbors(dbscan.Ballots[i].Votes)
// 	}

// }

// FindNeighbors finds the neighbors of a given point
func (dbscan *DBScan) FindNeighbors(point []float64) []float64 {
	


}

// ExpandCluster
func (dbscan *DBScan) ExpandCluster() {}