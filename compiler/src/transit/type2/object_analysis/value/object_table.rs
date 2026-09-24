use std::collections::BTreeMap;

use super::*;

#[derive(Clone, Debug)]
struct Assoc<T>(Vec<(ValueNode, T)>);

impl<T> Default for Assoc<T> {
    fn default() -> Self {
        Self(Vec::new())
    }
}

impl<T> Assoc<T> {
    fn get(&self, vn: &ValueNode) -> Option<&T> {
        self.0
            .iter()
            .filter_map(|(a, b)| if a == vn { Some(b) } else { None })
            .next()
    }

    fn get_mut(&mut self, vn: &ValueNode) -> Option<&mut T> {
        self.0
            .iter_mut()
            .filter_map(|(a, b)| if a == vn { Some(b) } else { None })
            .next()
    }

    fn insert(&mut self, vn: ValueNode, mut t: T) -> Option<T> {
        if let Some(x) = self.get_mut(&vn) {
            std::mem::swap(&mut t, x);
            Some(t)
        } else {
            self.0.push((vn, t));
            None
        }
    }

    fn remove(&mut self, vn: &ValueNode) -> Option<T> {
        let idx = self.0.iter().position(|(x, _)| x == vn)?;
        Some(self.0.remove(idx).1)
    }
}

#[derive(Clone, Debug)]
pub struct ObjectTable<T>(BTreeMap<ObjectId, Assoc<T>>, usize);

impl<T> ObjectTable<T> {
    pub fn new() -> Self {
        Self(BTreeMap::new(), 0)
    }

    pub fn contains(&self, object: &Object) -> bool {
        self.get(object).is_some()
    }

    pub fn get(&self, object: &Object) -> Option<&T> {
        self.0
            .get(&object.id())
            .map(|assoc| assoc.get(object.node()))
            .flatten()
    }

    pub fn get_mut(&mut self, object: &Object) -> Option<&mut T> {
        self.0
            .get_mut(&object.id())
            .map(|assoc| assoc.get_mut(object.node()))
            .flatten()
    }

    pub fn insert(&mut self, object: Object, t: T) -> Option<T> {
        let r = self
            .0
            .entry(object.id())
            .or_default()
            .insert(object.node, t);
        if r.is_none() {
            self.1 += 1;
        }
        r
    }

    pub fn remove(&mut self, object: &Object) -> Option<T> {
        let r = self
            .0
            .get_mut(&object.id())
            .map(|assoc| assoc.remove(object.node()))
            .flatten();

        if r.is_some() {
            self.1 -= 1;
        }

        r
    }

    pub fn len(&self) -> usize {
        self.1
    }

    pub fn iter<'a>(&'a self) -> impl Iterator<Item = (Object, &'a T)> + 'a {
        self.0.iter().flat_map(|(oid, assoc)| {
            assoc
                .0
                .iter()
                .map(move |(vn, x)| (Object::new(*oid, vn.clone()), x))
        })
    }

    pub fn keys<'a>(&'a self) -> impl Iterator<Item = Object> + 'a {
        self.0.iter().flat_map(|(oid, assoc)| {
            assoc
                .0
                .iter()
                .map(move |(vn, _)| Object::new(*oid, vn.clone()))
        })
    }

    pub fn values<'a>(&'a self) -> impl Iterator<Item = &'a T> + 'a {
        self.0
            .values()
            .flat_map(|assoc| assoc.0.iter().map(|(_, x)| x))
    }

    pub fn iter_mut<'a>(&'a mut self) -> impl Iterator<Item = (Object, &'a mut T)> + 'a {
        self.0.iter_mut().flat_map(|(oid, assoc)| {
            assoc
                .0
                .iter_mut()
                .map(move |(vn, x)| (Object::new(*oid, vn.clone()), x))
        })
    }

    pub fn map<T2>(self, mut f: impl FnMut(T) -> T2) -> ObjectTable<T2> {
        ObjectTable(
            self.0
                .into_iter()
                .map(|(oid, assoc)| {
                    (
                        oid,
                        Assoc(assoc.0.into_iter().map(|(vn, t)| (vn, f(t))).collect()),
                    )
                })
                .collect(),
            self.1,
        )
    }
}

impl<T> std::ops::Index<&Object> for ObjectTable<T> {
    type Output = T;
    fn index(&self, index: &Object) -> &Self::Output {
        self.get(index).unwrap()
    }
}

impl FromIterator<Object> for ObjectTable<()> {
    fn from_iter<T: IntoIterator<Item = Object>>(iter: T) -> Self {
        let mut r = ObjectTable::new();
        for o in iter {
            r.insert(o, ());
        }
        r
    }
}

impl<X> FromIterator<(Object, X)> for ObjectTable<X> {
    fn from_iter<T: IntoIterator<Item = (Object, X)>>(iter: T) -> Self {
        let mut r = ObjectTable::new();
        for (o, x) in iter {
            r.insert(o, x);
        }
        r
    }
}

#[derive(Debug, Clone)]
pub struct ObjectBijection<T> {
    forward: ObjectTable<T>,
    backward: BTreeMap<T, Object>,
}

impl<T> ObjectBijection<T> {
    pub fn new() -> Self {
        Self {
            forward: ObjectTable::new(),
            backward: BTreeMap::new(),
        }
    }
}

impl<T> ObjectBijection<T>
where
    T: Clone + Ord,
{
    /// Returns whether some mapping was overwritten
    pub fn insert(&mut self, object: Object, t: T) -> bool {
        let mut overwritten = false;

        if let Some(t_old) = self.forward.get(&object) {
            self.backward
                .remove(t_old)
                .expect("curresponding item to exist in backward mapping");
            overwritten = true;
        };

        if let Some(object_old) = self.backward.get(&t) {
            self.forward
                .remove(object_old)
                .expect("curresponding object to exist in forward mapping");
            overwritten = true;
        }

        self.forward.insert(object.clone(), t.clone());
        self.backward.insert(t.clone(), object.clone());

        overwritten
    }

    /// `insert` with overwritting forbiddened
    pub fn insert_checked(&mut self, object: Object, t: T)
    where
        T: std::fmt::Debug,
    {
        if let Some(t_old) = self.forward.get(&object) {
            panic!(
                "insertion failed: {:?} already forward mapped to {:?}",
                &object, t_old
            );
        };

        if let Some(object_old) = self.backward.get(&t) {
            panic!(
                "insertion failed: {:?} already backward mapped to {:?}",
                &t, object_old
            );
        }

        self.forward.insert(object.clone(), t.clone());
        self.backward.insert(t.clone(), object.clone());
    }

    pub fn remove_forward(&mut self, object: &Object) -> Option<T> {
        self.forward.remove(object).map(|t| {
            self.backward.remove(&t);
            t
        })
    }

    pub fn remove_backward(&mut self, t: &T) -> Option<Object> {
        self.backward.remove(t).map(|object| {
            self.forward.remove(&object);
            object
        })
    }

    pub fn get_forward(&self, object: &Object) -> Option<&T> {
        self.forward.get(object)
    }

    pub fn get_backward(&self, t: &T) -> Option<&Object> {
        self.backward.get(t)
    }

    pub fn len(&self) -> usize {
        self.forward.len()
    }

    pub fn is_empty(&self) -> bool {
        self.forward.len() == 0
    }

    pub fn iter<'a>(&'a self) -> impl Iterator<Item = (Object, &'a T)> + 'a {
        self.forward.iter()
    }

    pub fn contains_forward(&self, object: &Object) -> bool {
        self.forward.contains(object)
    }
}
