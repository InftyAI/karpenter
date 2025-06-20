/*
Copyright The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package test

import (
	"fmt"

	"github.com/imdario/mergo"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	llmazcoreapi "github.com/inftyai/llmaz/api/core/v1alpha1"
)

type OpenModelOptions struct {
	metav1.ObjectMeta

	Flavors []llmazcoreapi.Flavor
}

func OpenModel(overrides ...OpenModelOptions) *llmazcoreapi.OpenModel {
	options := OpenModelOptions{}
	for _, opts := range overrides {
		if err := mergo.Merge(&options, opts, mergo.WithOverride); err != nil {
			panic(fmt.Sprintf("Failed to merge options: %s", err))
		}
	}

	return &llmazcoreapi.OpenModel{
		ObjectMeta: ObjectMeta(options.ObjectMeta),
		Spec: llmazcoreapi.ModelSpec{
			InferenceConfig: &llmazcoreapi.InferenceConfig{
				Flavors: options.Flavors,
			},
		},
	}
}
