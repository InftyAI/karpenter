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

package scheduling

import (
	"context"
	"fmt"
	"strings"

	"github.com/awslabs/operatorpkg/serrors"
	"github.com/samber/lo"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/client-go/kubernetes/scheme"
	"k8s.io/klog/v2"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"

	"sigs.k8s.io/karpenter/pkg/utils/pretty"

	llmazcoreapi "github.com/inftyai/llmaz/api/core/v1alpha1"
	llmazinferenceapi "github.com/inftyai/llmaz/api/inference/v1alpha1"
)

func init() {
	// Add support for llmaz CRDs.
	utilruntime.Must(llmazcoreapi.AddToScheme(scheme.Scheme))
	utilruntime.Must(llmazinferenceapi.AddToScheme(scheme.Scheme))
}

func NewModelInference(kubeClient client.Client) *ModelInference {
	return &ModelInference{kubeClient: kubeClient}
}

type ModelInference struct {
	kubeClient client.Client
}

func (m *ModelInference) Inject(ctx context.Context, pod *v1.Pod) error {
	flavors, err := m.getInferenceFlavors(ctx, pod)
	if err != nil {
		return err
	}

	kept, rejected := lo.FilterReject(flavors, func(flavor llmazcoreapi.Flavor, _ int) bool {
		return len(flavor.NodeSelector) > 0
	})
	if len(rejected) > 0 || len(kept) == 0 {
		return nil
	}

	if pod.Spec.Affinity == nil {
		pod.Spec.Affinity = &v1.Affinity{}
	}
	if pod.Spec.Affinity.NodeAffinity == nil {
		pod.Spec.Affinity.NodeAffinity = &v1.NodeAffinity{}
	}
	if pod.Spec.Affinity.NodeAffinity.RequiredDuringSchedulingIgnoredDuringExecution == nil {
		pod.Spec.Affinity.NodeAffinity.RequiredDuringSchedulingIgnoredDuringExecution = &v1.NodeSelector{}
	}
	if len(pod.Spec.Affinity.NodeAffinity.RequiredDuringSchedulingIgnoredDuringExecution.NodeSelectorTerms) == 0 {
		pod.Spec.Affinity.NodeAffinity.RequiredDuringSchedulingIgnoredDuringExecution.NodeSelectorTerms = []v1.NodeSelectorTerm{{}}
	}

	podCopy := pod.DeepCopy()
	pod.Spec.Affinity.NodeAffinity.RequiredDuringSchedulingIgnoredDuringExecution.NodeSelectorTerms = nil

	// Add the inference flavor requirements to the pod's node affinity. This causes it to be OR'd with every merged requirement,
	// so that relaxation employs our flavor requirements according to the orders of the merged flavors,
	// when no existing node, in-flight node claim, or node pool can satisfy the current flavor requirements.
	lo.ForEach(kept, func(flavor llmazcoreapi.Flavor, _ int) {
		matchExpressions := lo.MapToSlice(flavor.NodeSelector, func(key string, value string) v1.NodeSelectorRequirement {
			return v1.NodeSelectorRequirement{
				Key:      key,
				Operator: v1.NodeSelectorOpIn,
				Values:   []string{value},
			}
		})
		// We add our inference requirement to every node selector term.  This causes it to be AND'd with every existing
		// requirement so that relaxation won't remove our inference requirement.
		nodeSelectorTermsCopy := podCopy.Spec.Affinity.NodeAffinity.RequiredDuringSchedulingIgnoredDuringExecution.DeepCopy().NodeSelectorTerms
		for i := 0; i < len(nodeSelectorTermsCopy); i++ {
			nodeSelectorTermsCopy[i].MatchExpressions = append(nodeSelectorTermsCopy[i].MatchExpressions, matchExpressions...)
		}
		log.FromContext(ctx).
			WithValues("Pod", klog.KObj(pod)).
			V(1).Info(fmt.Sprintf("adding requirements derived from pod's inference flavor %q, %s", flavor.Name, matchExpressions))
		pod.Spec.Affinity.NodeAffinity.RequiredDuringSchedulingIgnoredDuringExecution.NodeSelectorTerms = append(pod.Spec.Affinity.NodeAffinity.RequiredDuringSchedulingIgnoredDuringExecution.NodeSelectorTerms, nodeSelectorTermsCopy...)
	})

	log.FromContext(ctx).
		WithValues("Pod", klog.KObj(pod)).
		V(1).Info(fmt.Sprintf("adding requirements derived from pod's inference flavors, %s", pretty.Concise(pod.Spec.Affinity.NodeAffinity.RequiredDuringSchedulingIgnoredDuringExecution)))

	return nil
}

func (m *ModelInference) getInferenceFlavors(ctx context.Context, pod *v1.Pod) ([]llmazcoreapi.Flavor, error) {
	modelName, ok := pod.Labels[llmazcoreapi.ModelNameLabelKey]
	if !ok {
		// Ignore the pod that is not created via llmaz's inference service.
		return nil, nil
	}

	model := &llmazcoreapi.OpenModel{}
	if err := m.kubeClient.Get(ctx, types.NamespacedName{Name: modelName}, model); err != nil {
		return nil, fmt.Errorf("getting open model %q, %w", modelName, err)
	}
	modelFlavors := lo.FromPtrOr(model.Spec.InferenceConfig, llmazcoreapi.InferenceConfig{}).Flavors

	serviceFlavorRawStr, ok := pod.Annotations[llmazinferenceapi.InferenceServiceFlavorsAnnoKey]
	if !ok {
		// Not all inference pods specify the inference service flavors.
		return modelFlavors, nil
	}

	modelFlavorMap := lo.SliceToMap(modelFlavors, func(flavor llmazcoreapi.Flavor) (llmazcoreapi.FlavorName, llmazcoreapi.Flavor) {
		return flavor.Name, flavor
	})

	var result []llmazcoreapi.Flavor
	for _, flavorNameVal := range strings.Split(serviceFlavorRawStr, ",") {
		flavor, ok := modelFlavorMap[llmazcoreapi.FlavorName(flavorNameVal)]
		if !ok {
			return nil, fmt.Errorf("unknown service inference flavor %q", flavorNameVal)
		}
		result = append(result, flavor)
	}
	return result, nil
}

func (m *ModelInference) ValidateInferenceFlavors(ctx context.Context, pod *v1.Pod) (err error) {
	modelName, ok := pod.Labels[llmazcoreapi.ModelNameLabelKey]
	if !ok {
		// Ignore the pod that is not created via llmaz's inference service.
		return nil
	}

	model := &llmazcoreapi.OpenModel{}
	if err := m.kubeClient.Get(ctx, types.NamespacedName{Name: modelName}, model); err != nil {
		return serrors.Wrap(fmt.Errorf("failed to validate open model, %w", err), "OpenModel", klog.KRef("", modelName))
	}

	serviceFlavorRawStr, ok := pod.Annotations[llmazinferenceapi.InferenceServiceFlavorsAnnoKey]
	if !ok {
		// Not all inference pods specify the inference service flavors.
		return nil
	}

	// Get all flavors from the model and check if the service flavors are valid.
	allFlavors := lo.SliceToMap(
		lo.FromPtrOr(model.Spec.InferenceConfig, llmazcoreapi.InferenceConfig{}).Flavors,
		func(flavor llmazcoreapi.Flavor) (llmazcoreapi.FlavorName, llmazcoreapi.Flavor) {
			return flavor.Name, flavor
		},
	)
	unknownFlavors := lo.Reject(strings.Split(serviceFlavorRawStr, ","), func(flavor string, _ int) bool {
		return lo.HasKey(allFlavors, llmazcoreapi.FlavorName(flavor))
	})

	if len(unknownFlavors) > 0 {
		err = serrors.Wrap(fmt.Errorf("unknown service inference flavors, %v", unknownFlavors), "OpenModel", klog.KRef("", modelName))
		return err
	}
	return nil
}
